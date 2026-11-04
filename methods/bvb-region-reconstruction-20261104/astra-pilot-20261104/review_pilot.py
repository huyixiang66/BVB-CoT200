"""Present the six-question run as local raw-frame and reconstruction pairs."""
from html import escape
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from reconstruct_regions import read
from review_inputs import STYLE, contact

NAMES={'vsi/1585':'冰箱、沙发与水壶','vsi/2607':'数门','sti/6':'浴垫与马桶','sti/8':'9.4–13.1 秒相机轨迹',
       'mmsi/question_0645':'洗手池与办公桌','mmsi/question_0734':'126 秒相机朝向与浴室区域'}
LABELS={'bath mat':'浴垫','toilet':'马桶','sink':'洗手池','desk':'办公桌','bathroom':'浴室区域',
        'door':'门','refrigerator':'冰箱','sofa':'沙发','kettle':'水壶'}
STATUS={'ready':'已生成局部场景','empty':'未生成目标或墙体','invalid_response':'场景格式未通过','missing_response':'未收到重建结果'}


def photo(path,title):
    return f'<figure><img src="{path}" alt="{escape(title,quote=True)}"><figcaption>{escape(title)}</figcaption></figure>'


def main():
    index=read(HERE/'rendered/manifest.json')['questions']
    by_case={q['case']:q for q in index}
    style=STYLE+'''.maps figure{margin:0;min-width:0}.maps img{display:block;width:100%}.maps{grid-template-columns:repeat(auto-fit,minmax(min(100%,285px),1fr))}
nav{display:flex;gap:10px 16px;flex-wrap:wrap;margin:16px 0}.legend span{display:inline-block;margin-right:12px;font-weight:650}
details{padding:10px 0;border-top:1px solid #e4e8ed}.group{margin:20px 0}.group .maps{margin-top:7px}.overview{padding:16px;background:white;border:1px solid #d7dde3;border-radius:8px}'''
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
          '<title>三个场景 · 局部重建小规模对照</title>',f'<style>{style}</style>',
          '<h1>三个场景 · 局部重建小规模对照</h1>',
          '<p>指定 Astra 子代理读取每组原帧和逐图位姿，生成局部 JSON，再由学长的资产编译器拼入同一坐标系。',
          '物体候选来自 Grounding DINO Tiny。几何＋外观采用已存档的 0.75 预设，并按类别分组。</p>',
          '<nav>'+''.join(f'<a href="#{case.replace("/","-")}">{case.upper()} {title}</a>' for case,title in NAMES.items())+'</nav>',
          '<div class="overview"><strong>VSI 参照</strong> · 原有 BVB 全场景<div class="maps">',
          photo('../reference/top.png','已有 BVB 俯视'),photo('../reference/oblique.png','已有 BVB 斜俯视'),'</div></div>']
    for case,title in NAMES.items():
        q=by_case[case];target=HERE/'rendered'/case
        seen=sorted({s for g in q['groups'] for s in g.get('target_labels',[])})
        missing=[x for x in q['question']['objects'] if x not in seen]
        page.append(f'<section id="{case.replace("/","-")}"><h2>{escape(case.upper())} · {title}</h2>')
        page.append(f'<p>{escape(q["question"]["question"])}</p>')
        if case=='vsi/1585':
            page.append('<p>厨房冰箱候选来自 Tiny 的 <code>built-in refrigerator</code> 提示词诊断。此题手动限定了已知厨房时段补入候选。</p>')
            page.append('<p style="color:#a24227">三个沙发片段各自生成了资产，当前拼合仍有重复和遮挡，相关位置需要人工核对。</p>')
        if case=='vsi/2607':
            page.append('<p style="color:#a24227">各组生成的门分别着色，跨组重复的门假设仍在场景中。</p>')
        if case=='sti/8':
            page.append('<p>本题局部结构没有被确认，当前展示查询区间的轨迹、起止和朝向。</p>')
        if seen:
            page.append('<p>生成目标 · '+escape('、'.join(LABELS.get(x,x) for x in seen))+'</p>')
        if missing:
            page.append('<p style="color:#a24227">当前缺少 '+escape('、'.join(LABELS.get(x,x) for x in missing))+'</p>')
        if q['trajectory']:
            rows=q['trajectory']['rows']
            page.append(f'<p class="legend"><span style="color:#d63d31">红线：轨迹</span><span style="color:#18783b">绿球：起点 {rows[0]["timestamp_s"]:.2f}s</span><span style="color:#2559a6">蓝球：终点 {rows[-1]["timestamp_s"]:.2f}s</span><span style="color:#a65610">橙箭头：相机朝向</span>'
                        +('<span style="color:#6931a4">紫色：126 秒题目指定机位和朝向</span>' if case.startswith('mmsi') else '')+'</p>')
            page.append('<small>轨迹使用缓存的预测位姿，保留 LingBot 原生尺度。</small>')
        if (target/'top.png').exists():
            page.append('<div class="maps">'+photo(f'rendered/{case}/top.png','拼合俯视')+photo(f'rendered/{case}/oblique.png','拼合斜俯视')+'</div>')
            page.append(f'<p><a href="rendered/{case}/scene.blend">下载 Blender 场景</a></p>')
        page.append('<details><summary>逐组对照 · 原视频帧与局部重建</summary>')
        for group in q['groups']:
            gid=group['id'];raw=HERE/case/gid;out=target/'groups'/gid
            contact(raw,raw/'contact.jpg')
            status=STATUS[group['status']]
            page.append(f'<div class="group"><strong>{escape(gid)}</strong> · {group["frames"]} 张 · {status}<div class="maps">')
            page.append(photo(f'{case}/{gid}/contact.jpg','送给 Astra 的原帧，红框为 Tiny 候选'))
            if (out/'oblique.png').exists():
                page.append(photo(f'rendered/{case}/groups/{gid}/oblique.png','局部斜俯视'))
            page.append('</div></div>')
        page.append('</details></section>')
    page.append('</html>')
    (HERE/'review.html').write_text('\n'.join(page),encoding='utf-8')


if __name__=='__main__':
    main()
