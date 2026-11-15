"""Self-contained V4 delivery with V3 comparison and single-page gallery."""
import argparse,hashlib,html,json,shutil,math,re
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bvb_v3.single_page_delivery import render

def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def timecode(t):return f'{int(t)//60:02d}:{t%60:06.3f}'

def main():
    ap=argparse.ArgumentParser()
    for name in ['result','v3-delivery','cache','lingbot','out']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False);summary=read(a.result/'summary.json');groups=read(a.result/'groups.json');gm={g['group_id']:g for g in groups};old=read(a.v3_delivery/'manifest.json');oldby={r['id']:r for r in old['questions']};qs=read(a.result/'questions.json');cand=read(a.cache/'candidates.json');byframe={c['source_frame_index']:c for c in cand}
    poses=np.load(a.lingbot/'camera-poses.npz')['c2w'];oldvideo=a.v3_delivery/old['video']['path'];vp=out/'01-original-video'/oldvideo.name;vp.parent.mkdir();shutil.copy2(oldvideo,vp);assert sha(vp)==old['video']['sha256']
    qdir=out/'02-questions';qdir.mkdir();save(qdir/'questions.json',qs);shutil.copy2(a.v3_delivery/'02-questions/questions.txt',qdir/'questions.txt');per=out/'03-per-question';per.mkdir();method=out/'04-method-and-validation';method.mkdir()
    for name in ['config.json','summary.json','groups.json','association-audit.json','identity-hypotheses.json','anchor-geometry.json']:shutil.copy2(a.result/name,method/name)
    for name in ['select_frames.py','report.py','test_selection.py']:shutil.copy2(Path(__file__).parent/name,method/name)
    catalog=[];fileproof=[];font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',17);comparison=[];grouphtml={}
    for number,record in enumerate(summary['questions'],1):
        q=record['question'];qid=q['id'];folder=per/f'{number:02d}-q{qid}';folder.mkdir();fd=folder/'frames';fd.mkdir();gs=[gm[k] for k in record['group_ids']];oldentry=oldby[qid];oldfolder=a.v3_delivery/oldentry['folder'];positions=[]
        for i,fid in enumerate(record['source_frame_indices'],1):
            c=byframe[fid];source=a.cache/c['frame'];name=f'{i:03d}__{c["timestamp_s"]:010.3f}s__frame{fid:07d}.jpg';shutil.copy2(source,fd/name);assert sha(fd/name)==c['frame_sha256']
            uses=[{'object':g['object'],'group_id':g['group_id'],'scope':g['scope'],'reason':r['selection_reason'],'status':r['status'],'source_peaks':r['source_peaks'],'quality_score':r['quality_score'],'view_basis':r['view_basis']} for g in gs for r in g['selected'] if r['source_frame_index']==fid];context=any(u['status']=='context_fallback' for u in uses);evidence=any(u['status']!='context_fallback' for u in uses)
            positions.append({'number':i,'source_frame_index_zero_based':fid,'source_frame_number_one_based':fid+1,'timestamp_s':c['timestamp_s'],'timecode':timecode(c['timestamp_s']),'image':'frames/'+name,'sha256':c['frame_sha256'],'objects':list(dict.fromkeys(u['object'] for u in uses)),'uses':uses,'evidence_status':'mixed_evidence_and_context' if context and evidence else ('context_only' if context else 'detection_backed')});fileproof.append({'path':str((fd/name).relative_to(out)),'sha256':c['frame_sha256']})
        save(folder/'question.json',q);shutil.copy2(oldfolder/'question.txt',folder/'question.txt');save(folder/'sampling-positions.json',{'version':'v5','question_id':qid,'source_video':'../../'+str(vp.relative_to(out)),'source_video_fps':30,'frames':positions,'group_ids':record['group_ids'],'policy':'counting separate hypotheses' if q['question_type']=='object_counting' else 'shared per-question-object budget','comparison_v3':record['v3_unique_frames'],'physical_instance_count':None})
        (folder/'sampling-positions.txt').write_text('序号\t时间\t秒\t原视频帧索引(0-based)\t物体\t状态\t原因\n'+'\n'.join(f'{f["number"]}\t{f["timecode"]}\t{f["timestamp_s"]:.6f}\t{f["source_frame_index_zero_based"]}\t{", ".join(f["objects"])}\t{f["evidence_status"]}\t'+ '; '.join(u['reason'] for u in f['uses']) for f in positions)+'\n')
        save(folder/'input-frames.json',{'question':q,'version':'v5','frames':[{'source_frame_index':f['source_frame_index_zero_based'],'timestamp_s':f['timestamp_s'],'image':f['image'],'sha256':f['sha256'],'selection_uses':f['uses']} for f in positions],'chronological':True,'deduplicated':True})
        for name in ['dino-bars.png','dino-scores.json']:shutil.copy2(oldfolder/name,folder/name)
        prior=read(oldfolder/'sampling-positions.json')['frames'];fig,ax=plt.subplots(figsize=(16,2+.7*len(q['objects'])))
        for oi,o in enumerate(q['objects']):
            before=[f['timestamp_s'] for f in prior if o in f['objects']];ax.vlines(before,oi-.28,oi-.08,color='#c6cbd2',lw=1.3)
            for g in gs:
                if g['object']!=o:continue
                for r in g['selected']:ax.vlines(r['timestamp_s'],oi+.03,oi+.28,color='#d38b18' if r['status']=='context_fallback' else '#168957',lw=2)
        ax.set_yticks(range(len(q['objects'])),['keyboard (piano)' if o=='keyboard' else o for o in q['objects']]);ax.invert_yaxis();ax.set_xlim(-1,238);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Original video seconds');ax.set_title(f'Q{qid}: V3 {record["v3_unique_frames"]} -> V4 {record["unique_frames"]} unique frames\ngrey upper ticks: V3 / green lower: V4 detection-backed / orange: V4 context');fig.tight_layout();fig.savefig(folder/'sampling-positions.png',dpi=140);plt.close(fig)
        fig=plt.figure(figsize=(11,7));ax=fig.add_subplot(111,projection='3d');C=poses[:,:3,3];ax.plot(C[:,0],C[:,1],C[:,2],color='#bbb',lw=.7,alpha=.5)
        for gi,g in enumerate(gs):
            color=plt.get_cmap('tab10')(gi%10);rs=g['selected'];p=np.array([r['camera_center'] for r in rs]);directions=np.array([r['view_direction'] for r in rs]);ax.scatter(p[:,0],p[:,1],p[:,2],color=color,s=28,label=g['group_id']);ax.quiver(p[:,0],p[:,1],p[:,2],directions[:,0],directions[:,1],directions[:,2],length=.12,normalize=True,color=color,alpha=.6)
        ax.set_xlabel('LingBot X');ax.set_ylabel('LingBot Y');ax.set_zlabel('LingBot Z');ax.set_title(f'Q{qid} selected camera locations / observation directions\nRaw corrected LingBot coordinates; arbitrary scale, not Blender alignment');ax.legend(fontsize=7,loc='upper left');fig.tight_layout();fig.savefig(folder/'lingbot-views.png',dpi=140);plt.close(fig)
        rows=[];cards=[]
        for g in gs:
            rows.append(f'<tr><td>{html.escape(g["object"])}'+('（计数候选）' if g['scope']=='counting_hypothesis' else '')+f'</td><td>{len(g["peak_ids"])}</td><td>{len(g["candidates"])}</td><td>{g["selected_count"]}</td><td>{g["context_count"]}</td></tr>')
            rs=g['selected'];canvas=Image.new('RGB',(1280,math.ceil(len(rs)/4)*285),'white');draw=ImageDraw.Draw(canvas)
            for j,r in enumerate(rs):
                im=Image.open(a.cache/byframe[r['source_frame_index']]['frame']).convert('RGB');d=ImageDraw.Draw(im)
                if r['box_xyxy']:d.rectangle(r['box_xyxy'],outline='#168957',width=3)
                im=im.resize((320,240));x=j%4*320;y=j//4*285;canvas.paste(im,(x,y+45));draw.text((x+5,y+3),f'{g["object"]} | {r["timestamp_s"]:.2f}s',font=font,fill='black');draw.text((x+5,y+24),r['status'],font=font,fill='#a66312' if r['status']=='context_fallback' else '#176b45')
            pic=g['group_id']+'.jpg';canvas.save(folder/pic,quality=92);cards.append(pic)
        grouphtml[qid]='<h3>跨波峰预算</h3><p>'+('计数题：身份未确认的候选分别保留至少4张，表中候选数量不是实际门数。' if q['question_type']=='object_counting' else '非计数题：同一题目物体跨峰共享预算，最低4张。只有明确同屏分离的候选才预留多个实例各自的最低覆盖。')+'</p><table><tr><th>物体</th><th>原波峰数</th><th>候选帧数</th><th>V4保留</th><th>上下文</th></tr>'+''.join(rows)+'</table><h3>LingBot 机位与观察方向</h3><p class="legend">这是估计位姿；箭头可能使用框中心视线代理，不能据此宣称完成精确物体定位。</p><img class="chart" loading="lazy" src="'+str(folder.relative_to(out))+'/lingbot-views.png" alt="LingBot机位">'
        comparison.append({'id':qid,'type':q['question_type'],'v3_frames':record['v3_unique_frames'],'v4_frames':record['unique_frames'],'reduction':record['v3_unique_frames']-record['unique_frames'],'context_slots':record['context_slots']});catalog.append({'number':number,'id':qid,'question':q['question'],'objects':q['objects'],'folder':str(folder.relative_to(out)),'peaks':len(gs),'frame_slots':record['selected_slots'],'unique_frames':len(positions),'context_slots':record['context_slots']})
    save(out/'manifest.json',{'version':'v5','scene':summary['scene'],'video':{**old['video'],'path':str(vp.relative_to(out))},'questions':catalog,'images':fileproof,'per_question_image_copies':len(fileproof),'global_unique_source_frames':summary['global_unique_frames'],'comparison':comparison,'source_result':str(a.result.resolve())});save(out/'comparison-v3-v4.json',comparison)
    render(out);page=(out/'index.html').read_text();page=page.replace('十题自适应抽帧 · 全部内容','V5 · LingBot 自适应抽帧').replace('3db0a1c8f3：十题自适应抽帧','V4 · 3db0a1c8f3：LingBot 辅助选帧').replace(' 个采样波峰',' 个预算组').replace('红三角为保留锚点','红三角为 V3 候选锚点，最终 V4 选择见下图').replace('绿色：检测支持；橙色：未确认的上下文；紫色：同帧兼有两种用途；红色：锚点。','灰色上排：V3；绿色下排：V4 检测支持；橙色下排：V4 上下文补位。')
    intro=f'V4：非计数题每个题目物体跨波峰共享预算，至少4张，默认最多8张；明确同屏分离候选各预留4张。计数题保留身份未确认的候选覆盖。全局源图由V3的290张减为{summary["global_unique_frames"]}张；跨题复用后本页展示{len(fileproof)}张。{summary["context_slots"]}个预算帧位为未确认的上下文补位。未使用SAM2、未重新推理LingBot或DINO，未运行重建或QA。'
    page=re.sub(r'<p class="muted">1fps发现波峰.*?</p>','<p class="muted">'+intro+'</p>',page,flags=re.S)
    table='<h3>同十题 V3 / V4 对照</h3><table><tr><th>题目</th><th>V3图片</th><th>V4图片</th><th>减少</th></tr>'+''.join(f'<tr><td><a href="#q{r["id"]}">Q{r["id"]}</a></td><td>{r["v3_frames"]}</td><td>{r["v4_frames"]}</td><td>{r["reduction"]}</td></tr>' for r in comparison)+'</table><p class="legend">计数题没有为了减帧而强行合并疑似实例。几何覆盖增益为启发式，尚未验证QA或重构质量是否提高。</p>'
    page=page.replace('<video id="source-video"',table+'<video id="source-video"',1)
    for qid,block in grouphtml.items():page=page.replace(f'<h3 id="q{qid}-frames">',block+f'<h3 id="q{qid}-frames">',1)
    (out/'index.html').write_text(page)
    (out/'README.txt').write_text(intro+'\n\n打开index.html，原视频、十题、DINO条形图、V3/V4抽帧位置、LingBot机位与全部实际图片均在同一页。每题input-frames.json为按时间去重的重构输入清单。\n\n重要限制：本场景真实实例身份尚未完全解决。非计数题按题目词语共享预算，不代表已证明全部波峰来自同一物体。计数题则单独保留未知身份候选，不能用候选数当答案。位置为原始LingBot单位，没有使用旧BVB相机配准。稀疏框内点可能混入背景，无法证明定位误差下降。最低4张不保证4个独立视角；不足有效目标帧时上下文补位仍明确标记。\n\n首轮严格关联试验保留了82个身份假设、275张唯一图片，未解决重复预算，因此最终将身份假设审计与预算分配分开；计数题按用户进一步说明单独处理。所有试验均复用模型缓存。\n')
    # Policy, exact-frame and file verification, including each observed-instance reservation.
    for g in groups:
        selected={r['source_frame_index'] for r in g['selected']};assert len(selected)>=4
        required=set()
        for reserve in g['instance_reservations']:
            assert len(set(reserve['source_frame_indices']))==4 and set(reserve['source_frame_indices'])<=selected;required.update(reserve['source_frame_indices'])
        assert len(selected)<=max(8,len(required))
        for r in g['selected']:
            assert byframe[r['source_frame_index']]['timestamp_s']==r['timestamp_s'] and sha(a.cache/byframe[r['source_frame_index']]['frame'])==r['frame_sha256']
    for q in summary['questions']:
        expected='counting_hypothesis' if q['question']['question_type']=='object_counting' else 'non_counting_object'
        assert all(gm[k]['scope']==expected for k in q['group_ids'])
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.ids=set()
        def handle_starttag(self,tag,attrs):
            d=dict(attrs)
            if 'id' in d:self.ids.add(d['id'])
            for k in ['href','src','data-image']:
                if d.get(k):self.links.append(d[k])
    parser=Links();parser.feed(page);broken=[]
    for l in parser.links:
        u=urlsplit(l)
        if u.scheme or u.netloc:continue
        if u.path and not (out/unquote(u.path)).exists():broken.append(l)
        if not u.path and u.fragment and u.fragment not in parser.ids:broken.append(l)
    assert not broken,broken
    validation={'questions':10,'all_object_or_counting_hypothesis_minima_four':True,'all_same_frame_instance_reservations_preserved':True,'all_budgets_within_max_or_reservation_override':True,'counting_policy_separate':True,'exact_original_frame_pose_alignment':True,'raw_video_sha256_verified':True,'delivered_images_sha256_verified':len(fileproof),'html_links_checked':len(parser.links),'broken_links':broken,'sam2_used':False,'new_model_inference':False,'qa_run':False};save(out/'validation.json',validation);save(a.result/'validation.json',validation)
    from .annotate_delivery import annotate
    annotate(out,a.result,a.cache,a.lingbot)
    from .candidate_gallery import update
    update(out,a.result,a.cache)
    print(json.dumps({'comparison':comparison,'validation':read(out/'validation.json')},indent=2))

if __name__=='__main__':main()
