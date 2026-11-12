from pathlib import Path
import os,json,struct,subprocess,time,hashlib,csv
import numpy as np
b=Path.home()/'c4_migration/project';r=b/'pymaf_final/final_release';rt=b/'pi_build/runtime';out=Path(os.environ.get('BINPARAM_BENCHMARK_OUT', Path.home()/'c4_migration/precision_retry_20261109'));out.mkdir(exist_ok=True)
pack=r.parent/'dataset/3dpw_board_200';data=(pack/(pack/'manifest.txt').read_text().splitlines()[0].split()[1]).read_bytes()
assets={'smpl':r/'source/data/smpl_neutral_f32.bin','dmap':r/'source/data/smpl_dmap_431x6890.bin','mean':r/'source/data/pymaf_mean_params_f32.bin','smpl-onnx':r/'models/smpl_ort/smpl_neutral_opset16.onnx'}
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',PYMAF_ORT_DISABLE_SPIN='1',PYMAF_CAP_DOWNSAMPLE_THREADS='1',BOLT_BNN_ZERO_PADDING_FIX='1',BOLT_BNN_NUMERIC_ZERO_FIX='0')
for k in ('BOLT_TIMING_FILE','BOLT_DIAG_DIR','PYMAF_PHASE_LOG'):env.pop(k,None)
env['LD_LIBRARY_PATH']=':'.join([str(rt/'cap_release_engine'),str(rt/'lib_parallel_border')]+sorted({str(p.parent) for p in (b/'pi_build/bolt').rglob('*.so')}))
def read(f,n):
 x=bytearray()
 while len(x)<n:
  a=f.read(n-len(x))
  if not a:raise RuntimeError('incomplete RPC')
  x.extend(a)
 return bytes(x)
runs=[];models={}
for rep,order in enumerate([('fp32','fp16','bnn'),('fp16','bnn','fp32'),('bnn','fp32','fp16')],1):
 for label in order:
  paths=dict(assets)
  for n in ('feature','stage0','stage1','stage2'):
   paths[n]=r/'models/fp32'/(n+'.bolt') if label=='fp32' else (b/'pi_build/precision_models'/(n+'_f16.bolt') if label=='fp16' else r/'versions/c4_20261020_biasfix/models/bnn'/(n+'.bolt'))
  models[label]={k:dict(path=str(v),bytes=v.stat().st_size,sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in paths.items()}
  cmd=['taskset','-c','0-3',str(rt/'rpc_cap4')]
  for k,v in paths.items():cmd+=['--'+k,str(v)]
  cmd+=['--smpl-threads','4']; rows=[];hashes=[]
  with (out/(label+str(rep)+'.log')).open('w') as log:
   p=subprocess.Popen(cmd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
   try:
    assert struct.unpack_from('<I',read(p.stdout,24))[0]==0x50595244
    for i in range(35):
     p.stdin.write(data);p.stdin.flush();head=read(p.stdout,104);assert struct.unpack_from('<I',head)[0]==0x50595253
     payload=read(p.stdout,20899*4);assert np.isfinite(np.frombuffer(payload,dtype='<f4')).all()
     hashes.append(hashlib.sha256(payload).hexdigest())
     if i>=5:rows.append(struct.unpack_from('<11d',head,16)[0])
    status=(Path('/proc')/str(p.pid)/'status').read_text();peak=next(int(x.split()[1])/1024 for x in status.splitlines() if x.startswith('VmHWM:'))
    p.stdin.close();assert p.wait(timeout=30)==0
   finally:
    if p.poll() is None:p.kill();p.wait()
  assert len(set(hashes))==1
  runs.append(dict(config=label,round=rep,timings_ms=rows,peak_rss_mib=peak,output_sha256=hashes[0],temperature_c=float(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000))
  (out/'runs.json').write_text(json.dumps(runs,indent=2));(out/'models.json').write_text(json.dumps(models,indent=2));os.sync();print(label,rep,'mean',np.mean(rows),'peak',peak,flush=True)
summary={}
for label in ('fp32','fp16','bnn'):
 rr=[x for x in runs if x['config']==label];ts=[t for x in rr for t in x['timings_ms']];assert len({x['output_sha256'] for x in rr})==1
 summary[label]=dict(n=len(ts),mean_ms=float(np.mean(ts)),p95_ms=float(np.percentile(ts,95)),model_mib=sum(models[label][k]['bytes'] for k in ('feature','stage0','stage1','stage2'))/1024**2,peak_rss_mib=max(x['peak_rss_mib'] for x in rr))
for v in summary.values():v['speedup']=summary['fp32']['mean_ms']/v['mean_ms']
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'protocol.json').write_text(json.dumps(dict(warmup=5,loops=30,rounds=3,threads=4,affinity='0-3',input_sha256=hashlib.sha256(data).hexdigest(),environment={k:v for k,v in env.items() if k.startswith(("OMP_","PYMAF_","BOLT_","LD_LIBRARY_PATH"))},boundary='resident inference including final SMPL; excludes I/O preprocessing display',size_scope='four Bolt models only'),indent=2));os.sync();print(json.dumps(summary,indent=2),flush=True)
