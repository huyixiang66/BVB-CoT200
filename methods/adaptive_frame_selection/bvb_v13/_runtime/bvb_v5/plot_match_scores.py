"""Plot observed aggressive-policy pair scores; missing scores stay missing."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--delivery',type=Path,required=True)
    a=ap.parse_args()
    source=a.result/'peak-association.json'
    pairs=json.loads(source.read_text());cfg=json.loads((a.result/'config.json').read_text());threshold=cfg['threshold']
    valid=[p for p in pairs if p['multi_view_score'] is not None]
    values=np.array([p['multi_view_score'] for p in valid])
    missing=len(pairs)-len(valid);passing=int((values>=threshold).sum())
    veto=sum(p['multi_view_score']>=threshold and p['same_frame_distinct_evidence'] for p in valid)
    supported=sum(p['merge_supported'] for p in pairs)
    assert len(values)+missing==len(pairs)
    checks=[.5,.6,.65,.7,.75,.8,.85,.9]
    stats={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
           'policy':cfg.get('mode','conservative'),'score_definition':'best two disjoint frame matches; minimum of the two cosine similarities',
           'total_same_category_peak_pairs':len(pairs),'valid_scored_pairs':len(values),'missing_pairs':missing,
           'threshold':threshold,'at_or_above_threshold':passing,'above_threshold_same_frame_veto':veto,
           'direct_merge_supported_pairs':supported,'median':float(np.median(values)),
           'quantiles':{str(q):float(np.quantile(values,q)) for q in [.1,.25,.5,.75,.9]},
           'threshold_sweep_score_only':[{'threshold':t,'pairs':int((values>=t).sum()),'percent_of_valid':float(100*(values>=t).mean())} for t in checks]}
    out=a.result/'score-distribution';out.mkdir(exist_ok=True)
    (out/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    font=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
    fig=plt.figure(figsize=(15,11),facecolor='#fafbfe');gs=fig.add_gridspec(2,2,height_ratios=[1,1.35],hspace=.45,wspace=.25)
    ax=fig.add_subplot(gs[0,0]);survival=fig.add_subplot(gs[0,1]);byclass=fig.add_subplot(gs[1,:])
    bins=np.linspace(0,1,21)
    hist,edges=np.histogram(values,bins=bins)
    ax.bar(edges[:-1],hist,width=np.diff(edges)*.92,align='edge',color=['#3478b8' if x<threshold-1e-8 else '#e69039' for x in edges[:-1]])
    ax.axvline(threshold,color='#be3939',lw=2,ls='--',label=f'当前阈值 {threshold:.2f}')
    ax.axvline(np.median(values),color='#293547',lw=1.5,ls=':',label=f'中位数 {np.median(values):.3f}')
    ax.set(xlim=(0,1),xlabel='双图匹配分数（余弦相似度的较低值）',ylabel='同类波峰对数量',title=f'{len(values)} 个有效分数的直方图')
    ax.legend(frameon=False,fontsize=10);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ticks=np.linspace(0,1,501);rates=np.array([100*(values>=t).mean() for t in ticks])
    survival.plot(ticks,rates,color='#3478b8',lw=2.3)
    survival.axvline(threshold,color='#be3939',lw=1.5,ls='--')
    rate_labels=[]
    for t in [.6,.65,.7,.8]:
        count=int((values>=t).sum());y=100*count/len(values)
        survival.scatter([t],[y],color='#e69039',s=27,zorder=3)
        rate_labels.append(f'≥ {t:.2f}：{count} 对 ({y:.1f}%)')
    survival.text(.57,.88,'\n'.join(rate_labels),transform=survival.transAxes,va='top',fontsize=11,
                  linespacing=1.7,bbox={'facecolor':'white','edgecolor':'#dce3ed','boxstyle':'round,pad=.7'},zorder=5)
    survival.set(xlim=(0,1),ylim=(-2,103),xlabel='假设的匹配阈值',ylabel='达到阈值的比例（仅有效分数）',title='阈值降低后，有多少波峰对达到分数要求？')
    survival.grid(alpha=.15)
    categories=sorted({p['object'] for p in valid},key=lambda o:-np.median([p['multi_view_score'] for p in valid if p['object']==o]))
    data=[np.array([p['multi_view_score'] for p in valid if p['object']==o]) for o in categories]
    byclass.boxplot(data,orientation='horizontal',patch_artist=True,widths=.45,showfliers=False,
                         boxprops={'facecolor':'#dce8f5','edgecolor':'#3478b8'},medianprops={'color':'#163b63','linewidth':2},
                         whiskerprops={'color':'#8395a8'},capprops={'color':'#8395a8'})
    rng=np.random.default_rng(0)
    for i,vals in enumerate(data,1):
        byclass.scatter(vals,i+rng.uniform(-.13,.13,len(vals)),s=13,alpha=.5,c=['#3478b8' if v<threshold else '#e69039' for v in vals],edgecolors='none')
    byclass.axvline(threshold,color='#be3939',lw=1.6,ls='--')
    byclass.set_yticks(range(1,len(categories)+1),[f'{o}  (n={len(v)})' for o,v in zip(categories,data)])
    byclass.invert_yaxis();byclass.set(xlim=(0,1),xlabel='双图匹配分数',title='按物体类别分布：点是波峰对，箱体是中间 50%，竖线是中位数')
    byclass.grid(axis='x',alpha=.15)
    fig.suptitle('DINOv2 积极归并版 · 同类波峰对匹配分数分布',fontsize=20,x=.5,y=.98)
    fig.text(.5,.935,f'全部 {len(pairs)} 对  |  有效 {len(values)} 对  |  无双图匹配 {missing} 对（不记作 0 分）  |  ≥ {threshold:.2f}：{passing} 对（{passing/len(values):.1%}）',ha='center',fontsize=12,color='#42526a')
    fig.text(.06,.025,f'达到阈值的 {passing} 对中，{veto} 对被“同帧独立物体”否决，{supported} 对直接匹配通过。\n分母均为 {len(values)} 个有效分数；这是波峰对分布，不是物体数量、身份正确率或最终归并组数。类别仅展示有有效分数者。',fontsize=10,color='#42526a')
    fig.subplots_adjust(left=.17,right=.97,top=.88,bottom=.10)
    fig.savefig(out/'match-score-distribution.png',dpi=160,facecolor=fig.get_facecolor())
    fig.savefig(out/'match-score-distribution.svg',facecolor=fig.get_facecolor());plt.close(fig)
    dest=a.delivery/'06-score-distribution';shutil.copytree(out,dest,dirs_exist_ok=True)
    shutil.copy2(Path(__file__),dest/'plot_match_scores.py')
    import re
    page=a.delivery/'index.html';text=page.read_text();text=re.sub(r'<!-- match-score-distribution-start -->.*?<!-- match-score-distribution-end -->','',text,flags=re.S)
    block=f'''<!-- match-score-distribution-start --><section id="match-score-distribution"><h2>匹配分数分布</h2><p>{len(pairs)} 对同类波峰中 {len(values)} 对有有效分数，{missing} 对无法形成双图匹配。中位数 {np.median(values):.3f}；达到 {threshold:.2f} 的只有 {passing} 对（{passing/len(values):.1%}）。其中 {veto} 对被同帧独立约束否决。</p><a href="06-score-distribution/match-score-distribution.png" target="_blank"><img class="chart" loading="lazy" src="06-score-distribution/match-score-distribution.png" alt="匹配分数分布直方图、阈值通过率与分类分布"></a><p><a href="06-score-distribution/match-score-distribution.svg">矢量图 SVG</a> · <a href="06-score-distribution/stats.json">完整统计</a></p></section><!-- match-score-distribution-end -->'''
    marker='<section id="q';at=text.index(marker);text=text[:at]+block+text[at:];page.write_text(text)
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
