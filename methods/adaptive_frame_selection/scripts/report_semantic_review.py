"""Publish Astra veto decisions without deleting the original clustering evidence."""
import argparse,html,json,shutil,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bvb_v6.semantic_review import read,write,export,sha
LABEL={'accept':'通过','reject':'拒绝：误检','uncertain':'无法确认','pending':'未完成'}
COLOR={'accept':'#217b42','reject':'#bf2637','mixed':'#9757ba','uncertain':'#a36b0b','pending':'#64748b'}
ADMISSION={'semantic_accept':'保留（确认目标）','only_candidate':'保留（唯一候选）','comparison_best':'保留（对比择优）','not_selected':'不选用'}
def esc(v):return html.escape(str(v))
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--delivery',type=Path,required=True);a=p.parse_args()
    s=export(a.source);m=read(a.source/'manifest.json')
    if sha(a.delivery/'selection.json')!=m['selection_sha256']:raise ValueError('Delivery does not match reviewed selection')
    review_dir='semantic-review-clean' if m.get('image_policy')=='clean-full-and-context-crop-v2' else 'semantic-review'
    out=a.delivery/review_dir;out.mkdir(exist_ok=True)
    counts=Counter();token=Counter();jobdata={}
    for gid in m['jobs']:
        folder=a.source/'jobs'/gid;j=read(folder/'input.json');review=read(folder/'review.json') if (folder/'review.json').exists() else {'status':'pending','evidence':[]}
        counts[review['status']]+=1;jobdata[gid]=(j,review)
        if (folder/'request.json').exists():
            usage=read(folder/'request.json').get('usage') or {}
            token.update({k:usage.get(k,0) for k in ['prompt_tokens','completion_tokens','total_tokens']})
        dest=out/'jobs'/gid;dest.mkdir(parents=True,exist_ok=True)
        for name in ['input.json','review.json','request.json','response.txt']:
            if (folder/name).exists():shutil.copy2(folder/name,dest/name)
        for e in j['evidence']:
            for key in ['full','crop']:shutil.copy2(folder/e[key],dest/e[key])
    for name in ['reviewed-selection.json','summary.json','manifest.json','system-prompt.txt']:
        shutil.copy2(a.source/name,out/name)
    if (a.source/'image-validation.json').exists():shutil.copy2(a.source/'image-validation.json',out/'image-validation.json')
    coverage=(a.source/'coverage-policy.json').exists();coverage_stats={};comparison_token=Counter();comparisons=[]
    if coverage:
        for name in ['coverage-policy.json','coverage-summary.json']:shutil.copy2(a.source/name,out/name)
        shutil.copytree(a.source/'coverage',out/'coverage',dirs_exist_ok=True)
        coverage_stats=read(a.source/'coverage-summary.json')
        cm=read(a.source/'coverage/manifest.json')
        for cid in cm['comparisons']:
            folder=a.source/'coverage'/cid
            if (folder/'request.json').exists():
                usage=read(folder/'request.json').get('usage') or {}
                comparison_token.update({k:usage.get(k,0) for k in ['prompt_tokens','completion_tokens','total_tokens']})
            if (folder/'result.json').exists():
                r=read(folder/'result.json');j=read(folder/'input.json')
                rows=''.join(f'<tr><td>{v["rank"]}</td><td>{esc(v["candidate_id"])}</td><td>{esc(v["reason"])}</td></tr>' for v in sorted(r['ranking'],key=lambda x:x['rank']))
                comparisons.append(f'<section><h2>{esc(j["object"])} 候选对比</h2><p>选中 {esc(r["selected_id"])}，置信程度 {esc(r["confidence"])}。{esc(r["reason"])}</p><p>仍有不确定性：{esc(r["uncertainty"])}</p><table><tr><th>排名</th><th>候选审核 ID（对应下方各组）</th><th>理由</th></tr>{rows}</table><a href="coverage/{cid}/result.json">对比原始结果</a></section>')
    groupcounts=Counter();table=[];sections=[]
    base=a.delivery/'index-before-semantic-review.html'
    if not base.exists():shutil.copy2(a.delivery/'index.html',base)
    page=base.read_text()
    for qid,q in s['questions'].items():
        groups=q['arms'][m['arm']];parts=[];qc=Counter()
        admissions={g['group_id']:g['semantic_review'].get('admission',{}) for g in q['arms'].get('semantic_selected',[])+q.get('coverage_excluded',[])}
        start=f'<section id="q{qid}">';left,rest=page.split(start,1);section,right=rest.split('</section>',1)
        pre,robust=section.split('data-panel="roi_robust"',1)
        for g in groups:
            gid=m['references'][qid][g['group_id']];j,r=jobdata[gid];status=r['status'];qc[status]+=1;groupcounts[status]+=1
            verdicts={e['evidence_id']:e for e in r['evidence']};cards=[]
            for e in j['evidence']:
                v=verdicts.get(e['evidence_id'],{'verdict':'pending','observed_object':'','reason':'等待审核'})
                cards.append(f'<article><h4>{e["timestamp_s"]:.2f}s · {esc(e["peak_id"])}</h4><div class="pictures"><img src="jobs/{gid}/{e["full"]}" loading="lazy"><img src="jobs/{gid}/{e["crop"]}" loading="lazy"></div><p><b>{esc(v["verdict"])}</b> · {esc(v["observed_object"])}：{esc(v["reason"])}</p></article>')
            anchor=f'q{qid}-{g["group_id"]}'
            admission=admissions.get(g['group_id'],{});decision=ADMISSION.get(admission.get('mode'),'')
            note=f'<p><b>{decision}</b>。{esc(admission.get("reason",""))} {esc(admission.get("uncertainty",""))}</p>' if decision else ''
            parts.append(f'<div id="{anchor}" class="group" style="border-color:{COLOR[status]}"><h3>{esc(g["group_id"])} · <span style="color:{COLOR[status]}">{LABEL[status]}</span></h3>'+note+f'<p>目标：{esc(j["intended_category"])} · 审核 ID：{gid} · 审核 {len(j["evidence"])} 个波峰代表框 · 同组跨题复用审核。<a href="jobs/{gid}/input.json">输入</a></p>'+''.join(cards)+'</div>')
            needle=f'<h4>{g["group_id"]} <small>'
            badge=f'<p class="semantic-review-badge" style="color:{COLOR[status]}"><b>{decision} · Astra 判断：{LABEL[status]}</b> · <a href="{review_dir}/index.html#{anchor}">查看判断、实际输入图与理由</a></p>'
            assert needle in robust
            robust=robust.replace(needle,badge+needle,1)
        page=left+start+pre+'data-panel="roi_robust"'+robust+'</section>'+right
        cs=coverage_stats.get(qid,{})
        cells=''.join(f'<td>{cs.get(k,0)}</td>' for k in ['selected','confirmed','forced_single','compared']) if coverage else ''
        table.append(f'<tr><td><a href="#q{qid}">{qid}</a></td>'+''.join(f'<td>{qc[k]}</td>' for k in LABEL)+cells+'</tr>')
        missing=q.get('coverage_missing_categories',q.get('semantic_missing_categories',[]))
        note='<p><b>尚无选中候选的目标：</b>'+esc(', '.join(missing))+'。不能据此回答该物体不存在或数量为零。</p>' if missing else '<p>每个目标类别均已保留至少一个候选；覆盖保留不等于语义确认。</p>' if coverage else ''
        sections.append(f'<section id="q{qid}"><h2>Q{qid}</h2><p>{esc(q["question"]["question"])}</p>'+note+''.join(parts)+'</section>')
    policy='保留确认组；唯一候选不拒绝；多候选无确认组时对比择优，确保各目标至少保留一个候选。语义判断与是否保留分开显示。' if coverage else '只有通过组进入后续重构。'
    block=f'<p class="note"><a href="{review_dir}/index.html">Astra 语义审核与候选选择</a>。Astra 输入为无框原帧和无框裁剪；本页带框图仅供人工核对。{policy}</p>'
    page=page.replace('</h1>','</h1>'+block,1);(a.delivery/'index.html').write_text(page)
    css='body{font:16px system-ui;background:#f3f5f9;color:#243148;margin:auto;max-width:1320px;padding:24px}header,section{background:white;padding:24px;margin:20px 0;border-radius:12px}p{line-height:1.6}.group{border-left:5px solid;padding:16px;margin:24px 0;background:#f6f7fa}article{padding:10px;border-top:1px solid #ddd}.pictures{display:grid;grid-template-columns:2fr 1fr;gap:12px}.pictures img{width:100%}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}video{width:640px;max-width:100%}a{color:#2256a0}'
    header='<h1>Astra 聚类语义审核</h1><p>官方 gpt-6-astra / high。对每个波峰的代表框同时查看原帧与局部裁剪，检测分数不作为语义真值。只要有一个框确认属于目标，就接受整组，包括同时存在误检或不确定框的情况；没有确认目标且全部明确误检才拒绝，其余保留待核查。通过表示存在目标证据，不代表每个框都正确，也不证明整组是同一实例。</p>'
    if review_dir=='semantic-review-clean':
        header+='<p><b>本次诊断图片完全无 BBX、标签或分数叠加。</b>完整帧与缓存原图逐字节一致，局部图从干净原图裁剪、等比例补边到 384×384，并以 PNG 保存。候选位置仅通过文字坐标描述。以下图片就是本轮实际发送给 Astra 的输入。</p>'
        if (out/'image-validation.json').exists():header+='<p><a href="image-validation.json">无框输入像素校验</a></p>'
    header+=f'<p>{len(m["jobs"])} 次独立审核，对应 {sum(groupcounts.values())} 个题目聚类。状态：{esc(dict(groupcounts))}。API 返回输入 {token["prompt_tokens"]:,} / 输出 {token["completion_tokens"]:,} tokens（输出含推理）。未重构、未跑 QA。</p>'
    if coverage:
        total=sum(v['selected'] for v in coverage_stats.values())
        header+=f'<p><b>覆盖保留规则：</b>{policy} 当前共保留 {total} 个题目聚类。优先比较 uncertain 候选，若全部被判误检则比较所有候选；没有任何检测候选时记录缺失。额外对比输入 {comparison_token["prompt_tokens"]:,} / 输出 {comparison_token["completion_tokens"]:,} tokens。</p><p><a href="coverage-summary.json">各题覆盖统计</a></p>'
    header+='<p><a href="../index.html">返回原始聚类和波形</a> · <a href="reviewed-selection.json">带审核门禁的选帧文件</a> · <a href="summary.json">结果汇总</a></p><video controls preload="metadata" src="../original-video.mp4"></video>'
    table='<table><tr><th>题号</th>'+''.join(f'<th>原判断：{v}</th>' for v in LABEL.values())+(''.join(f'<th>{v}</th>' for v in ['最终保留','确认目标','唯一候选保留','对比择优保留']) if coverage else '')+'</tr>'+''.join(table)+'</table>'
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Astra 聚类审核与覆盖选择</title><style>'+css+'</style><header>'+header+table+'</header>'+''.join(comparisons)+''.join(sections)+'</html>')
    write(out/'usage.json',dict(unique_reviews=len(m['jobs']),unique_status=dict(counts),question_group_status=dict(groupcounts),usage=dict(token),comparison_usage=dict(comparison_token)))
    # Portable media/link check, including cross-page section anchors.
    from html.parser import HTMLParser
    from urllib.parse import urlsplit,unquote
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.ids=set()
        def handle_starttag(self,tag,attrs):
            d=dict(attrs);self.links.extend(d[k] for k in ['src','href'] if d.get(k))
            if d.get('id'):self.ids.add(d['id'])
    for path in [out/'index.html',a.delivery/'index.html']:
        parser=Links();parser.feed(path.read_text())
        for link in parser.links:
            u=urlsplit(link)
            if u.scheme:continue
            target=(path.parent/unquote(u.path)) if u.path else path
            assert target.is_file(),target
            if u.fragment and target.suffix=='.html':
                dest=Links();dest.feed(target.read_text());assert u.fragment in dest.ids
    write(out/'validation.json',dict(status='passed',source_selection_sha256=m['selection_sha256'],
         groups=sum(groupcounts.values()),unique_reviews=len(m['jobs']),reviewed_selection_sha256=sha(out/'reviewed-selection.json'),
         original_groups_preserved=True,reconstruction_run=False))
    print(json.dumps(dict(counts=groupcounts,usage=token),ensure_ascii=False));print(out/'index.html')
if __name__=='__main__':main()
