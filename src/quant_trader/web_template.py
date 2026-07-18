"""Single-file frontend for the local experiment platform."""

# ruff: noqa: E501 -- preserving readable HTML/CSS/JavaScript source lines

WEB_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant Trader Lab</title>
<style>
:root{--bg:#f4f7f5;--card:#fff;--ink:#17201c;--muted:#68736d;--line:#dce4df;--green:#146c4a;--green2:#e8f4ee;--red:#a43b3b;--amber:#9a6410;--shadow:0 12px 32px #163b2912}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:28px 30px 24px;background:#10251c;color:#fff}header h1{margin:0;font-size:25px;letter-spacing:.2px}header p{margin:5px 0 0;color:#bcd0c5}
.layout{display:grid;grid-template-columns:330px minmax(0,1fr);gap:18px;max-width:1440px;margin:0 auto;padding:20px}.stack{display:grid;gap:16px;align-content:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}h2{font-size:16px;margin:0 0 14px}h3{font-size:14px;margin:16px 0 8px}
label{display:block;color:var(--muted);font-size:12px;margin:12px 0 5px}select,input{width:100%;padding:10px 11px;border:1px solid #cbd6d0;border-radius:9px;background:#fff;color:var(--ink);font:inherit}
button{border:0;border-radius:9px;padding:11px 14px;background:var(--green);color:#fff;font-weight:650;cursor:pointer;width:100%;margin-top:15px}button:disabled{opacity:.55;cursor:wait}
.hint{font-size:12px;color:var(--muted);margin:8px 0 0}.error{color:var(--red);font-size:12px;margin-top:8px}.hidden{display:none!important}
.runs{display:grid;gap:8px;max-height:420px;overflow:auto}.run{border:1px solid var(--line);border-radius:10px;padding:10px;cursor:pointer;background:#fff}.run:hover,.run.active{border-color:#5a9278;background:#f1f8f4}.run strong{display:block}.run small{color:var(--muted)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#999;margin-right:6px}.dot.running{background:#2a8f62;box-shadow:0 0 0 4px #2a8f6222}.dot.completed{background:#146c4a}.dot.partial{background:#c18520}.dot.failed{background:#b83d3d}.dot.queued{background:#c18520}
.hero{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.hero h2{font-size:21px;margin:0}.badge{padding:5px 9px;border-radius:999px;background:var(--green2);color:var(--green);font-size:12px;font-weight:700}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:10px;margin-top:16px}.metric{padding:13px;border:1px solid var(--line);border-radius:11px;background:#fafcfb}.metric span{display:block;font-size:11px;color:var(--muted)}.metric b{display:block;margin-top:4px;font-size:17px;overflow-wrap:anywhere}
.columns{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(320px,.9fr);gap:16px}.log{height:370px;overflow:auto;background:#132019;color:#d6e4dc;border-radius:10px;padding:12px;font:12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace}.line{display:grid;grid-template-columns:72px 70px 1fr;gap:8px;border-bottom:1px solid #ffffff0d;padding:3px 0}.line time{color:#8ba798}.line em{color:#a8cbb9;font-style:normal}.line span{white-space:pre-wrap;overflow-wrap:anywhere}
.empty{padding:36px 10px;text-align:center;color:var(--muted)}.result{display:grid;gap:10px;max-height:540px;overflow:auto}.panel{border:1px solid var(--line);border-radius:10px;padding:12px}.panel p{margin:5px 0;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line)}th{color:var(--muted)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f6;padding:10px;border-radius:8px;font-size:11px;max-height:260px;overflow:auto}
.contestants{display:grid;gap:5px;max-height:120px;overflow:auto}.check{display:flex;gap:8px;align-items:center;margin:0;color:var(--ink);font-size:12px}.check input{width:auto}.pulse{animation:pulse 1.5s infinite}@keyframes pulse{50%{opacity:.55}}
@media(max-width:950px){.layout,.columns{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header><h1>Quant Trader Lab</h1><p>规则回测与论文实验控制台 · 仅研究和纸面模拟</p></header>
<main class="layout">
  <aside class="stack">
    <section class="card"><h2>新建实验</h2>
      <form id="form">
        <label for="mode">实验模式</label><select id="mode">
          <option value="rules">规则基线</option><option value="trading-agents">TradingAgents 多 Agent</option>
          <option value="finmem">FinMem 记忆策略</option><option value="quanta-alpha">QuantaAlpha 因子挖掘</option><option value="alpha-arena">Alpha Arena 横向对比</option>
        </select>
        <label for="provider">执行引擎</label><select id="provider"><option value="rules">本地规则</option><option value="codex">本地 Codex</option><option value="minimax">MiniMax M3（国内）</option></select>
        <div id="reviewsBox" class="hidden"><label for="reviews">最多完整审核次数</label><input id="reviews" type="number" min="1" max="10" value="1"><p class="hint">上限用于控制耗时和模型额度。</p></div>
        <div id="arenaBox" class="hidden"><label>参赛实验（可不选）</label><div id="contestants" class="contestants"></div></div>
        <button id="submit" type="submit">提交后台运行</button><div id="formError" class="error"></div>
      </form>
      <p class="hint">配置、行情路径由服务端固定。API Key 只读取当前终端环境变量。</p>
    </section>
    <section class="card"><h2>最近实验</h2><div id="runs" class="runs"><div class="empty">暂无实验</div></div></section>
  </aside>
  <section class="stack">
    <section id="overview" class="card"><div class="empty">在左侧提交实验，或选择一条历史任务。</div></section>
    <div class="columns">
      <section class="card"><h2>中间过程监控</h2><div id="log" class="log"><div class="empty">等待任务日志</div></div></section>
      <section class="card"><h2>实验效果</h2><div id="result" class="result"><div class="empty">任务完成后显示结果</div></div></section>
    </div>
  </section>
</main>
<script>
const api='api/runs', $=id=>document.getElementById(id);let selected=null,lastEventCount=0,loadedTerminal=null;
const names={rules:'规则基线','trading-agents':'TradingAgents','finmem':'FinMem','quanta-alpha':'QuantaAlpha','alpha-arena':'Alpha Arena',minimax:'MiniMax M3',codex:'Codex'};
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;return n}
function clear(n){n.replaceChildren()}
function fmt(v){if(v===null||v===undefined)return '—';if(typeof v==='number'){if(Math.abs(v)<=1&&v!==0)return (v*100).toFixed(2)+'%';return Number.isInteger(v)?String(v):v.toFixed(4)}return String(v)}
function when(s){if(!s)return '—';return new Date(s).toLocaleString('zh-CN',{hour12:false})}
function metric(root,label,value){const x=node('div',undefined,'metric');x.append(node('span',label),node('b',fmt(value)));root.append(x)}
function modeChanged(){const m=$('mode').value;const llm=['trading-agents','finmem','quanta-alpha'].includes(m);$('provider').value=llm?($('provider').value==='rules'?'codex':$('provider').value):'rules';$('provider').disabled=!llm;$('reviewsBox').classList.toggle('hidden',m!=='trading-agents');$('arenaBox').classList.toggle('hidden',m!=='alpha-arena')}
$('mode').addEventListener('change',modeChanged);modeChanged();
async function loadRuns(){try{const r=await fetch(api);if(!r.ok)return;const body=await r.json();renderRuns(body.runs||[]);renderContestants(body.runs||[]);if(!selected&&body.runs&&body.runs[0])selected=body.runs[0].id;if(selected&&selected!==loadedTerminal)await loadOne(selected)}catch(e){$('formError').textContent='无法连接本地服务'} }
function renderRuns(runs){const root=$('runs');clear(root);if(!runs.length){root.append(node('div','暂无实验','empty'));return}for(const run of runs){const item=node('div',undefined,'run'+(run.id===selected?' active':''));item.append(node('strong',(names[run.mode]||run.mode)+' · '+(names[run.provider]||run.provider)));const small=node('small');small.append(node('span',undefined,'dot '+run.status),document.createTextNode(run.status+' · '+when(run.created_at)));item.append(small);item.onclick=()=>{selected=run.id;loadedTerminal=null;lastEventCount=0;loadOne(run.id);renderRuns(runs)};root.append(item)}}
function renderContestants(runs){const root=$('contestants'),checked=new Set([...root.querySelectorAll('input:checked')].map(x=>x.value));clear(root);const usable=runs.filter(x=>['completed','partial'].includes(x.status)&&x.artifact_root&&['finmem','quanta-alpha'].includes(x.mode));if(!usable.length){root.append(node('span','完成 FinMem 或 QuantaAlpha 后可选择','hint'));return}for(const run of usable){const label=node('label',undefined,'check'),input=document.createElement('input');input.type='checkbox';input.value=run.id;input.checked=checked.has(run.id);label.append(input,document.createTextNode((names[run.mode]||run.mode)+' · '+run.id.slice(0,7)));root.append(label)}}
$('form').addEventListener('submit',async e=>{e.preventDefault();$('submit').disabled=true;$('formError').textContent='';const payload={mode:$('mode').value,provider:$('provider').value,max_reviews:Number($('reviews').value),contestant_ids:[...$('contestants').querySelectorAll('input:checked')].map(x=>x.value)};try{const r=await fetch(api,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const body=await r.json();if(!r.ok)throw new Error(body.error||'提交失败');selected=body.id;loadedTerminal=null;lastEventCount=0;await loadRuns()}catch(err){$('formError').textContent=err.message}finally{$('submit').disabled=false}});
async function loadOne(id){const r=await fetch(api+'/'+encodeURIComponent(id));if(!r.ok){selected=null;return}const run=await r.json();renderOverview(run);renderLog(run);renderResult(run);if(['completed','partial','failed'].includes(run.status))loadedTerminal=id;}
function renderOverview(run){const root=$('overview');clear(root);const hero=node('div',undefined,'hero'),left=node('div');left.append(node('h2',(names[run.mode]||run.mode)+' 实验'),node('p','任务 '+run.id+' · '+(names[run.provider]||run.provider),'hint'));hero.append(left,node('span',run.status,'badge'+(run.status==='running'?' pulse':'')));root.append(hero);const ms=node('div',undefined,'metrics');metric(ms,'创建时间',when(run.created_at));metric(ms,'开始时间',when(run.started_at));metric(ms,'结束时间',when(run.finished_at));metric(ms,'进程退出码',run.exit_code);root.append(ms);if(run.error)root.append(node('p',run.error,'error'))}
function renderLog(run){const root=$('log');const events=run.events||[];if(events.length===lastEventCount)return;clear(root);if(!events.length){root.append(node('div','等待任务日志','empty'));return}for(const ev of events){const row=node('div',undefined,'line');row.append(node('time',new Date(ev.at).toLocaleTimeString('zh-CN',{hour12:false})),node('em',ev.kind),node('span',ev.message));root.append(row)}lastEventCount=events.length;root.scrollTop=root.scrollHeight}
function renderResult(run){const root=$('result');clear(root);if(!run.result){root.append(node('div',run.status==='failed'?'运行失败，请查看左侧日志':'任务运行中，结果尚未生成','empty'));return}const value=run.result.details||run.result;if(run.mode==='alpha-arena')renderArena(root,value);else if(run.mode==='quanta-alpha')renderQuanta(root,value);else if(run.mode==='finmem')renderFinMem(root,value);else renderBacktest(root,value);const details=document.createElement('details');details.append(node('summary','查看原始结果 JSON'));const pre=node('pre',JSON.stringify(run.result,null,2));details.append(pre);root.append(details)}
function renderBacktest(root,v){const runs=v.runs||{result:v};for(const [name,run] of Object.entries(runs)){const title=node('h3',name),m=run.metrics||{},grid=node('div',undefined,'metrics');root.append(title);for(const [k,label] of Object.entries({total_return:'总收益',annualized_return:'年化收益',max_drawdown:'最大回撤',sharpe:'夏普比率',trade_count:'成交数',costs:'交易成本'}))if(k in m)metric(grid,label,m[k]);root.append(grid)}if(v.review_metadata){const p=node('div',undefined,'panel');p.append(node('strong','模型审核摘要'),node('p',JSON.stringify(v.review_metadata)));root.append(p)}}
function renderFinMem(root,v){renderBacktest(root,v);if(v.decision){const p=node('div',undefined,'panel');p.append(node('strong','最终记忆决策'),node('p',(v.decision.action||'—')+' · '+(v.decision.reason||'')));root.append(p)}if(Array.isArray(v.memory)){const p=node('div',undefined,'panel');p.append(node('strong','记忆记录：'+v.memory.length));root.append(p)}}
function renderQuanta(root,v){if(v.champion){const p=node('div',undefined,'panel');p.append(node('strong','优胜因子'),node('p',v.champion.expression||'—'));root.append(p)}for(const c of (v.candidates||[])){const p=node('div',undefined,'panel');p.append(node('strong',c.expression||'未命名候选'),node('p',c.rejection_reason?'拒绝：'+c.rejection_reason:'已通过安全校验'));root.append(p)}}
function renderArena(root,v){const table=node('table'),head=node('tr');for(const x of ['排名','实验','状态','总收益','最大回撤','成本'])head.append(node('th',x));table.append(head);for(const [i,row] of (v.leaderboard||[]).entries()){const tr=node('tr');for(const x of [i+1,row.name,row.status,fmt(row.total_return),fmt(row.max_drawdown),fmt(row.costs)])tr.append(node('td',x));table.append(tr)}root.append(table)}
setInterval(loadRuns,1000);loadRuns();
</script>
</body></html>"""
