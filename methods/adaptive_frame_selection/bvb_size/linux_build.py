"""Linux build adapter: isolate network, writable paths and home directories with bwrap."""
import argparse,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--virtual',required=True);p.add_argument('--blender',type=Path,required=True);p.add_argument('--probe',action='store_true');a=p.parse_args()
out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);binary=a.blender.resolve()
cmd=['bwrap','--unshare-all','--die-with-parent','--new-session']
for name in ['/usr','/bin','/sbin','/lib','/lib64','/etc']:
 if Path(name).exists():cmd+=['--ro-bind',name,name]
cmd+=['--proc','/proc','--dev','/dev','--tmpfs','/tmp','--ro-bind',str(binary.parent),str(binary.parent),'--bind',str(out),a.virtual,'--chdir',a.virtual,'--clearenv','--setenv','HOME',a.virtual,'--setenv','PATH','/usr/bin:/bin','--setenv','OMP_NUM_THREADS','2','--setenv','OPENBLAS_NUM_THREADS','2']
if a.probe:
 code="import pathlib; p=pathlib.Path('.'); (p/'probe-write.txt').write_text('ok'); print('ISOLATION_OK')"
 cmd+=['/usr/bin/python3','-c',code]
else:cmd += [str(binary),'--background','--disable-autoexec','--threads','2','--python-exit-code','1','--factory-startup','--python',a.virtual+'/generated.py','--python',a.virtual+'/validate.py','--',a.virtual]
r=subprocess.run(cmd,timeout=600);raise SystemExit(r.returncode)
