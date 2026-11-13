"""ODROID local camera frontend; one latest-frame slot and resident C++ backend."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from person_detector import PersonDetector
from pose_actions import ArmRaiseRecognizer

TIMING_NAMES = ['inference_ms', 'feature_ms', 'stage0_ms', 'stage1_ms',
                'stage2_ms', 'rotation6d_ms', 'smpl_feedback_ms', 'dmap_ms',
                'projection_ms', 'gridsample_ms', 'final_smpl_ms']
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
           16, 17, 18, 19, 20, 21]


def read_exact(stream, count):
    parts = bytearray()
    while len(parts) < count:
        block = stream.read(count - len(parts))
        if not block:
            raise RuntimeError('inference backend closed its output; see engine.log')
        parts.extend(block)
    return bytes(parts)


def memory_mib(pid):
    result = {}
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                result[line.split(':')[0]] = int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return result


def preprocess(frame, bbox, padding):
    h, w = frame.shape[:2]
    if bbox is None:
        side = min(h, w)
        x, y, bw, bh = (w-side)/2, (h-side)/2, side, side
    else:
        x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0 or x+bw <= 0 or y+bh <= 0 or x >= w or y >= h:
        raise ValueError('人体框无效或完全在图像外')
    cx, cy = x+bw/2, y+bh/2
    side = max(bw, bh) * padding
    left, top = cx-side/2, cy-side/2
    # Square crop with black padding outside the image; preserve body aspect ratio.
    scale = 224.0 / side
    matrix = np.array([[scale, 0, -left*scale], [0, scale, -top*scale]], np.float32)
    crop = cv2.warpAffine(frame, matrix, (224, 224), flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    chw = ((rgb-np.array([.485,.456,.406], np.float32)) /
           np.array([.229,.224,.225], np.float32)).transpose(2,0,1).copy()
    return chw, crop, [left, top, side, side]


def faces_from_smpl(path):
    with open(path, 'rb') as stream:
        if stream.read(8) != b'SMPLC001':
            raise ValueError('invalid SMPL binary')
        version, v, j, b, p = struct.unpack('<5I', stream.read(20))
        if version != 2:
            raise ValueError('OBJ export requires SMPL binary v2 with faces')
        face_count = struct.unpack('<I', stream.read(4))[0]
        stream.seek(4*(v*3 + v*3*b + p*v*3 + j*v + v*j + j), 1)
        return np.frombuffer(stream.read(face_count*3*4), dtype='<u4').reshape(-1,3)


def skeleton_panel(joints, axes, title):
    panel = np.full((320,280,3), 245, np.uint8)
    coords = joints[:, axes].copy()
    coords -= joints[0, axes]
    coords[:,1] *= -1
    extent = max(float(np.ptp(coords[:,0])), float(np.ptp(coords[:,1])), .5)
    coords -= (coords.min(0)+coords.max(0))/2
    pixels = (coords*(240/extent)+[140,175]).astype(int)
    for child, parent in enumerate(PARENTS):
        if parent >= 0:
            cv2.line(panel, tuple(pixels[parent]), tuple(pixels[child]), (160,90,20), 2)
    for pixel in pixels:
        cv2.circle(panel, tuple(pixel), 3, (30,60,200), -1)
    cv2.putText(panel, title, (12,25), cv2.FONT_HERSHEY_SIMPLEX,.55,(30,30,30),1)
    return panel


def save_snapshot(result, folder, faces=None):
    stamp = result['metadata']['frame_id']
    target = folder / f"snapshot_{time.time_ns()}_frame{stamp}"
    target.mkdir(parents=True)
    np.savez_compressed(target/'prediction.npz', **result['arrays'])
    if 'input_chw' in result:
        result['input_chw'].astype('<f4').tofile(target/'input_chw_f32.bin')
    (target/'metadata.json').write_text(json.dumps(result['metadata'],indent=2)+'\n')
    cv2.imwrite(str(target/'frame.jpg'), result['frame'])
    cv2.imwrite(str(target/'crop_224.jpg'), result['crop'])
    if faces is not None:
        with (target/'mesh.obj').open('w') as stream:
            for v in result['arrays']['vertices']:
                stream.write('v %.8f %.8f %.8f\n' % tuple(v))
            for face in faces:
                stream.write('f %d %d %d\n' % tuple(face+1))
    print('SAVED', target, flush=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--capture-fps', type=float, default=10)
    parser.add_argument('--duration', type=float, default=600, help='maximum runtime seconds, including no-person periods')
    parser.add_argument('--input-image', help='offline repeated-image smoke test; NOT a camera performance test')
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('X','Y','W','H'))
    parser.add_argument('--padding', type=float, default=1.2)
    parser.add_argument('--detector-model', required=True, help='YOLOv8n FP32 ONNX, fixed 320')
    parser.add_argument('--backend', help='Explicit resident RPC executable; verify its engine configuration separately')
    parser.add_argument('--backend-lib-dir', help='Libraries paired with the selected backend')
    parser.add_argument('--opencv-threads', type=int, default=4)
    parser.add_argument('--redetect-seconds', type=float, default=0,
                        help='0: detect each reconstruction; positive: short optical-flow tracking between detections')
    parser.add_argument('--select-roi', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--preview-only', action='store_true', help='show live image and inference telemetry, no skeleton or mesh rendering')
    parser.add_argument('--frames', type=int, default=0, help='0 means until stopped')
    parser.add_argument('--save-at', type=int, default=0, help='save one specified inference for testing')
    parser.add_argument('--save-obj', action='store_true')
    parser.add_argument('--review-on-save', action='store_true', help='S saves matched prediction and opens 3D viewer; Q in viewer returns to camera')
    parser.add_argument('--review-mode', choices=['mesh','overlay'], default='mesh', help='saved-result display used with --review-on-save')
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--action-config', help='JSON thresholds for four-class SMPL pose rules')
    args = parser.parse_args()
    if args.bbox is not None or args.select_roi:
        parser.error('automatic person detection is enabled; manual ROI is not used')
    cv2.setNumThreads(args.opencv_threads)
    if args.redetect_seconds < 0:
        parser.error('redetect interval must be nonnegative')
    detector = PersonDetector(args.detector_model, redetect_seconds=args.redetect_seconds)
    action_config = Path(args.action_config) if args.action_config else Path(__file__).with_name('pose_actions_config.json')
    actions = ArmRaiseRecognizer(action_config if action_config.exists() else None)
    if args.padding <= 0 or args.frames < 0 or args.duration <= 0 or args.capture_fps <= 0:
        parser.error('padding must be positive; frames must be nonnegative')
    if not args.headless and not os.environ.get('DISPLAY'):
        parser.error('set DISPLAY for HDMI display, or use --headless')
    root = Path(args.root)
    folder = Path(args.save_dir) if args.save_dir else root/'outputs'/f'camera_{time.time_ns()}'
    folder.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    review_pause = threading.Event()
    infer_idle = threading.Event()
    review_seconds = 0.0
    lock = threading.Lock()
    state = {'latest': None, 'result': None, 'error': None, 'completed': 0}
    test_frame = cv2.imread(args.input_image) if args.input_image else None
    if args.input_image and test_frame is None:
        raise RuntimeError('cannot read offline test image')
    cap = None
    if test_frame is None:
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.capture_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f'Cannot open /dev/video{args.device}; check USB camera connection')
    config = dict(vars(args))
    config['source'] = 'offline_repeated_image' if test_frame is not None else 'usb_camera'
    if cap is not None:
        config['negotiated_capture'] = {name: cap.get(prop) for name, prop in
            [('width', cv2.CAP_PROP_FRAME_WIDTH), ('height', cv2.CAP_PROP_FRAME_HEIGHT),
             ('fps', cv2.CAP_PROP_FPS), ('fourcc', cv2.CAP_PROP_FOURCC)]}
    (folder/'configuration.json').write_text(json.dumps(config, indent=2))

    def capture():
        index = 0
        try:
            while not stop.is_set():
                if test_frame is not None:
                    if stop.wait(1/args.capture_fps):
                        break
                    ok, frame = True, test_frame.copy()
                else:
                    ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('camera read failed')
                # One replaceable slot; no queue of unprocessed frames.
                with lock:
                    state['latest'] = (index, time.monotonic(), time.time(), frame)
                index += 1
        except Exception as error:
            state['error'] = str(error); stop.set()
        finally:
            if cap is not None:
                cap.release()

    capture_thread = threading.Thread(target=capture, daemon=True)
    capture_thread.start()
    process = None
    worker = None
    engine_log = None
    try:
        startup = time.monotonic()
        while state['latest'] is None and not stop.wait(.02):
            if time.monotonic()-startup > 10:
                raise RuntimeError('No decoded camera frame within 10 seconds')
        if stop.is_set():
            raise RuntimeError(state['error'])
        if args.select_roi:
            with lock:
                reference = state['latest'][3].copy()
            roi = cv2.selectROI('Select person; ENTER confirms', reference, False, False)
            cv2.destroyAllWindows()
            if roi[2] == 0 or roi[3] == 0:
                raise RuntimeError('ROI selection cancelled')
            args.bbox = list(roi)
        cmd = [args.backend or str(root/'bin/pymaf_camera_rpc')]
        paths = {'feature': root/'models/bnn/feature.bolt',
                 'stage0': root/'models/bnn/stage0.bolt',
                 'stage1': root/'models/bnn/stage1.bolt',
                 'stage2': root/'models/bnn/stage2.bolt',
                 'smpl': root/'source/data/smpl_neutral_f32.bin',
                 'dmap': root/'source/data/smpl_dmap_431x6890.bin',
                 'mean': root/'source/data/pymaf_mean_params_f32.bin',
                 'smpl-onnx': root/'models/smpl_ort/smpl_neutral_opset16.onnx'}
        for name, path in paths.items():
            cmd.extend(['--'+name, str(path)])
        cmd.extend(['--smpl-threads','4'])
        engine_log = (folder/'engine.log').open('w')
        engine_env = os.environ.copy()
        engine_env.update(PYMAF_ORT_DISABLE_SPIN='1', BOLT_BNN_ZERO_PADDING_FIX='1',
                          LD_LIBRARY_PATH=args.backend_lib_dir or str(root/'lib'))
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=engine_log, bufsize=0, env=engine_env)
        header = struct.unpack('<4I', read_exact(process.stdout,16))
        if header[:2] != (0x50595244,1):
            raise RuntimeError('bad backend startup response')
        initialization = struct.unpack('<d', read_exact(process.stdout,8))[0]
        print(f'READY initialization_ms={initialization:.2f} backend_pid={process.pid}',flush=True)

        def infer():
            last_id = -1
            session_begin = time.monotonic()
            try:
                with (folder/'timings.jsonl').open('w') as log:
                    while not stop.is_set():
                        if review_pause.is_set():
                            infer_idle.set()
                            stop.wait(.05)
                            continue
                        infer_idle.clear()
                        with lock:
                            item = state['latest']
                        if item[0] == last_id:
                            stop.wait(.005); continue
                        frame_id, received, wall, frame = item
                        detection_begin = time.monotonic()
                        bbox, detection_ms = detector.detect(frame)
                        if bbox is None:
                            actions.reset()
                            pending=folder/'latest_pose.pending.json'
                            pending.write_text(json.dumps({'metadata':{'status':'no_person'},'action':{'label':'UNKNOWN'}}))
                            pending.replace(folder/'latest_pose.json')
                            with lock:
                                state['result'] = None
                            log.write(json.dumps({'frame_id': frame_id, 'status': 'no_person',
                                                  'detection_ms': detection_ms})+'\n')
                            log.flush()
                            last_id = frame_id
                            stop.wait(.05)
                            continue
                        chw, crop, box = preprocess(frame,bbox,args.padding)
                        started = time.monotonic()
                        process.stdin.write(chw.tobytes())
                        # Raw pipes may perform short writes; use complete send.
                        # stdin uses a buffered wrapper below for full payload writes.
                        process.stdin.flush()
                        header = struct.unpack('<4I',read_exact(process.stdout,16))
                        if header[:3] != (0x50595253,1,20899):
                            raise RuntimeError('bad result header')
                        times = struct.unpack('<11d',read_exact(process.stdout,88))
                        values = np.frombuffer(read_exact(process.stdout,20899*4),'<f4').copy()
                        if not np.isfinite(values).all():
                            raise RuntimeError('nonfinite prediction')
                        arrays = {'pose6d':values[:144].reshape(24,6),
                                  'shape':values[144:154], 'camera':values[154:157],
                                  'joints':values[157:229].reshape(24,3),
                                  'vertices':values[229:].reshape(6890,3)}
                        now = time.monotonic()
                        memory = memory_mib(process.pid)
                        frontend = memory_mib(os.getpid())
                        metadata = dict(zip(TIMING_NAMES,times))
                        metadata.update(frame_id=frame_id, timestamp_unix=wall,
                            result_ready_monotonic=now,
                            status='person', detection_ms=detection_ms, detector_box_xywh=bbox,
                            source=config['source'],
                            localization_mode=detector.mode,
                            input_age_ms=(detection_begin-received)*1000,
                            preprocessing_ms=(started-detection_begin)*1000-detection_ms,
                            crop_box_xywh=box, queue_age_ms=(started-received)*1000,
                            frame_to_result_ms=(now-received)*1000,
                            rpc_ms=(now-started)*1000, skipped_frames=max(0,frame_id-last_id-1) if last_id>=0 else 0,
                            backend_rss_mib=memory.get('VmRSS'), backend_peak_mib=memory.get('VmHWM'),
                            frontend_rss_mib=frontend.get('VmRSS'),
                            coordinate_system='native SMPL model coordinates; meters; not world-space',
                            pose_format='24x6 PyMAF rotation6d (interleaved columns)',
                            input_normalization='RGB ImageNet mean/std',
                            initialization_ms=initialization)
                        metadata['output_fps'] = (state['completed']+1)/(now-session_begin)
                        metadata['latency_definition'] = 'decoded camera frame arrival to result; excludes sensor exposure and display'
                        # A clipped detection is a conservative warning, not proof of visibility.
                        bx,by,bw,bh = bbox
                        body_at_edge = by <= 1 or by+bh >= frame.shape[0]-1
                        fh,fw=frame.shape[:2]
                        severe_crop = by < .03*fh and by+bh > .97*fh and bw > .70*fw
                        action=actions.update(arrays['joints'], body_at_frame_edge=body_at_edge,
                                              body_severely_clipped=severe_crop)
                        metadata['action']=action
                        # Small atomic status file for later OLED/LED integration; mesh remains in memory.
                        latest = {'metadata':metadata, 'pose6d':arrays['pose6d'].tolist(),
                                  'action':action,
                                  'shape':arrays['shape'].tolist(), 'camera':arrays['camera'].tolist(),
                                  'joints':arrays['joints'].tolist()}
                        pending = folder/'latest_pose.pending.json'
                        pending.write_text(json.dumps(latest))
                        pending.replace(folder/'latest_pose.json')
                        result = {'arrays':arrays,'metadata':metadata,'frame':frame,'crop':crop,'input_chw':chw}
                        with lock:
                            state['result'] = result
                            state['completed'] += 1
                            count = state['completed']
                        log.write(json.dumps(metadata)+'\n'); log.flush()
                        print(f"FRAME {count} camera_id={frame_id} skipped={metadata['skipped_frames']} "
                              f"infer_ms={times[0]:.2f} frame_to_result_ms={metadata['frame_to_result_ms']:.2f} "
                              f"rss_mib={memory.get('VmRSS',0):.2f}",flush=True)
                        if count == args.save_at:
                            save_snapshot(result,folder,faces_from_smpl(paths['smpl']) if args.save_obj else None)
                        last_id = frame_id
                        if args.frames and count >= args.frames:
                            stop.set()
            except Exception as error:
                state['error'] = str(error); stop.set()

        # BufferedWriter guarantees full image writes even with a small pipe buffer.
        import io
        process.stdin = io.BufferedWriter(process.stdin)
        worker = threading.Thread(target=infer, daemon=True)
        worker.start()
        last_save_count = -1
        save_button = (660, 365, 940, 420)
        mouse_state = {'save_requested': False}
        window_name = 'ODROID BinParam local camera'

        def on_mouse(event, x, y, flags, userdata):
            bx1, by1, bx2, by2 = save_button
            if event == cv2.EVENT_LBUTTONUP and bx1 <= x <= bx2 and by1 <= y <= by2:
                mouse_state['save_requested'] = True

        if not args.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window_name, on_mouse)
        deadline = time.monotonic()+args.duration
        while not stop.is_set():
            if time.monotonic() >= deadline:
                stop.set()
                break
            with lock:
                item, result, count = state['latest'], state['result'], state['completed']
            if args.headless:
                # Touch save.request from SSH, or later connect a GPIO button to it.
                request = folder/'save.request'
                if request.exists() and result is not None:
                    request.unlink()
                    save_snapshot(result,folder,faces_from_smpl(paths['smpl']) if args.save_obj else None)
                stop.wait(.05); continue
            display = cv2.resize(item[3],(640,360))
            box = result['metadata']['crop_box_xywh'] if result is not None else [0,0,0,0]
            sx, sy = 640/item[3].shape[1], 360/item[3].shape[0]
            x,y,w,h = box
            cv2.rectangle(display,(int(x*sx),int(y*sy)),(int((x+w)*sx),int((y+h)*sy)),(0,230,0),2)
            canvas = np.zeros((480,1200,3),np.uint8)
            canvas[:360,:640] = display
            if result is not None and not args.preview_only:
                joints = result['arrays']['joints']
                canvas[:320,640:920] = skeleton_panel(joints,[0,1],'Predicted FRONT')
                canvas[:320,920:1200] = skeleton_panel(joints,[2,1],'Predicted SIDE')
                m = result['metadata']
                text = f"True-BNN | {m['inference_ms']:.1f} ms | {m['output_fps']:.2f} FPS | RSS {m['backend_rss_mib']:.1f} MiB"
                cv2.putText(canvas,text,(15,390),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1)
            if args.preview_only:
                if result is None:
                    text='Waiting for person / first inference'
                else:
                    m=result['metadata']
                    age=time.monotonic()-m['result_ready_monotonic']
                    text=f"BNN {m['inference_ms']:.0f} ms | Results {m['output_fps']:.2f}/s | age {age:.1f}s"
                    j=result['arrays']['joints']
                    cv2.putText(canvas,'SMPL pose features | subject left/right',(660,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1)
                    action=m.get('action',{})
                    cv2.putText(canvas,action.get('label','UNKNOWN'),(660,315),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,240,240),2)
                    required=action.get('required_predictions',3)
                    cv2.putText(canvas,f'Full body visible | hold {required} predictions',(660,345),cv2.FONT_HERSHEY_SIMPLEX,.43,(220,220,220),1)
                    f=action.get('features',{})
                    rows=[f"Arm lift L/R: {f.get('left_arm_lift_deg',0):.0f} / {f.get('right_arm_lift_deg',0):.0f} deg",
                          f"Wrist rel L/R: {f.get('left_wrist_torso',0):+.2f} / {f.get('right_wrist_torso',0):+.2f}",
                          f"Knee L/R: {f.get('left_knee_deg',0):.0f} / {f.get('right_knee_deg',0):.0f} deg",
                          f"Leg extension: {f.get('leg_extension_ratio',0):.2f}",
                          f"Candidate: {action.get('candidate','UNKNOWN')} {min(action.get('consecutive',0),required)}/{required}",
                          action.get('reason','UNKNOWN')]
                    for k,row in enumerate(rows):
                        cv2.putText(canvas,row,(660,80+k*32),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,240,180),1)
                cv2.putText(canvas,text,(15,390),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1)
            cv2.putText(canvas,('S: save + view 3D   Q: quit' if args.review_on_save else 'S: save prediction   Q: quit')+' | latest completed prediction',
                        (15,435),cv2.FONT_HERSHEY_SIMPLEX,.50,(230,230,230),1)
            bx1, by1, bx2, by2 = save_button
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (35, 125, 55), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (100, 240, 130), 2)
            button_text = 'Save + view 3D' if args.review_on_save else 'Save prediction'
            cv2.putText(canvas,button_text,(bx1+22,by1+39),cv2.FONT_HERSHEY_SIMPLEX,.68,(255,255,255),2)
            cv2.imshow(window_name,canvas)
            key = cv2.waitKey(15) & 255
            try:
                window_closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                window_closed = True
            if key in (ord('q'),27) or window_closed:
                stop.set()
                break
            save_requested = key == ord('s') or mouse_state['save_requested']
            if save_requested:
                mouse_state['save_requested'] = False
            if save_requested and result is not None and count != last_save_count:
                target = save_snapshot(result,folder,faces_from_smpl(paths['smpl']) if args.save_obj else None)
                last_save_count = count
                if args.review_on_save:
                    review_begin = time.monotonic()
                    review_pause.set()
                    try:
                        cv2.putText(canvas,'Opening saved 3D result...',(15,465),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,240,240),1)
                        cv2.imshow(window_name,canvas)
                        cv2.waitKey(1)
                        if not infer_idle.wait(timeout=10):
                            print('REVIEW_SKIPPED: inference did not become idle',flush=True)
                        else:
                            cv2.destroyWindow('ODROID BinParam local camera')
                            viewer = 'view_mesh_overlay.py' if args.review_mode == 'overlay' else 'view_saved_mesh.py'
                            code = subprocess.call([sys.executable,str(root/'tools'/viewer),
                                                    '--snapshot',str(target),'--root',str(root)])
                            print(f'REVIEW_CLOSED exit_code={code}; returning to camera',flush=True)
                    finally:
                        elapsed = time.monotonic()-review_begin
                        review_seconds += elapsed
                        deadline += elapsed
                        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                        cv2.setMouseCallback(window_name, on_mouse)
                        review_pause.clear()
        if state['error']:
            raise RuntimeError(state['error'])
    finally:
        stop.set()
        if worker is not None: worker.join(timeout=5)
        capture_thread.join(timeout=3)
        if process is not None:
            process.stdin.close()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate(); process.wait(timeout=5)
            if process.returncode:
                print('backend_exit_code',process.returncode)
        if engine_log is not None: engine_log.close()
        if not args.headless: cv2.destroyAllWindows()
    print('COMPLETED',state['completed'],flush=True)
    records = [json.loads(line) for line in (folder/'timings.jsonl').read_text().splitlines()]
    successful = [r for r in records if r.get('status') == 'person']
    measured = successful[3:]
    summary = {'source':config['source'], 'results':len(successful),
               'review_seconds':review_seconds,
               'result_rate_includes_review_pauses':bool(review_seconds),
               'no_person':sum(r.get('status') == 'no_person' for r in records),
               'warmup_results_excluded':min(3,len(successful)), 'measured_results':len(measured)}
    for key in ('detection_ms','preprocessing_ms','inference_ms','rpc_ms','input_age_ms','frame_to_result_ms'):
        if measured:
            values = [r[key] for r in measured]
            summary[key] = {'mean':float(np.mean(values)), 'p50':float(np.median(values)),
                            'p95':float(np.percentile(values,95))}
    if len(measured)>1:
        elapsed = measured[-1]['result_ready_monotonic']-measured[0]['result_ready_monotonic']
        summary['result_rate_fps'] = (len(measured)-1)/elapsed if elapsed>0 else None
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
