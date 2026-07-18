"""Fixed, dependency-free runtime dashboard page."""

# ruff: noqa: E501

DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TradingAgents 实时决策</title>
<style>
:root{color-scheme:light;--ink:#172033;--muted:#667085;--line:#d7deea;--blue:#2563eb;
--green:#15803d;--amber:#b45309;--red:#b91c1c;--paper:#fff;--wash:#f5f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"PingFang SC",sans-serif}
main{max-width:1180px;margin:auto;padding:24px}.top,.panel{background:var(--paper);border:1px solid
var(--line);border-radius:14px;box-shadow:0 8px 24px #1d293914}.top{display:flex;
justify-content:space-between;gap:16px;align-items:center;padding:18px 22px}.title{font-size:20px;
font-weight:750}.meta{color:var(--muted)}.status{font-weight:700;color:var(--blue)}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:14px;margin-top:14px}.panel{padding:18px}
.label{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:750}
.roles{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:12px}.role{appearance:none;
text-align:left;border:1px solid var(--line);border-radius:10px;background:white;padding:11px;min-height:67px;
cursor:pointer;color:inherit}.role:hover,.role.selected{border-color:var(--blue);box-shadow:0 0 0 2px #2563eb20}
.role.running{border-color:var(--blue);background:#eff6ff}.role.completed{border-color:#86efac;background:#f0fdf4}
.role.skipped{color:var(--muted);background:#f8fafc}.role.failed{border-color:#fca5a5;background:#fef2f2}
.role-name{font-weight:750}.role-note{font-size:12px;margin-top:4px;color:var(--muted)}
.detail{min-height:235px}.detail h2{font-size:18px;margin:8px 0}.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{border-radius:999px;background:#eef2ff;padding:3px 8px;font-size:12px}.list{margin:7px 0;padding-left:20px}
.safety{background:#fff7ed;border-color:#fdba74}.safety p{margin:8px 0}.decision{margin-top:14px;display:grid;
grid-template-columns:1fr 1fr;gap:14px}.decision h3{margin:6px 0}.empty{color:var(--muted)}
.bar{height:5px;background:#e5e7eb;border-radius:4px;overflow:hidden;margin-top:13px}.bar span{display:block;
height:100%;background:var(--blue);transition:width .3s}.footer{color:var(--muted);font-size:12px;margin-top:12px}
@media(max-width:800px){.grid,.decision{grid-template-columns:1fr}.roles{grid-template-columns:repeat(2,1fr)}
.top{align-items:flex-start;flex-direction:column}}
</style>
</head>
<body><main>
<section class="top"><div><div class="title" id="title">等待 TradingAgents 工作流</div>
<div class="meta" id="meta">本地只读 Dashboard</div></div><div class="status" id="status">准备中</div></section>
<div class="grid"><section class="panel"><div class="label">决策流程</div><div class="roles" id="roles"></div>
<div class="bar"><span id="progress" style="width:0%"></span></div></section>
<aside class="panel safety"><div class="label">硬性安全边界</div><p>✓ 只能处理规则选中的标的</p>
<p>✓ Agent 不能提高基础权重</p><p>✓ 失败时自动 reject / 0</p><p>✓ 仅研究与纸面交易</p></aside></div>
<div class="grid"><section class="panel detail"><div class="label">节点详情</div><div id="detail" class="empty">等待角色开始...</div></section>
<section class="panel"><div class="label">实时状态</div><div id="activity" class="empty">准备数据与模型...</div></section></div>
<div class="decision"><section class="panel"><div class="label">交易员建议</div><div id="proposal" class="empty">等待交易员</div></section>
<section class="panel"><div class="label">最终组合决策</div><div id="final" class="empty">等待风险评审</div></section></div>
<section class="panel" id="experiment-panel" hidden><div class="label">Paper Experiment 驾驶舱</div>
<div id="experiment-summary" class="empty">等待实验开始...</div><div id="experiment-detail"></div></section>
<div class="footer">页面只显示经过验证的结构化结论，不包含原始模型输出、Prompt 或凭证。</div>
</main><script>
const roleLabels={market_analyst:'市场分析',sentiment_analyst:'情绪分析',news_analyst:'新闻分析',
fundamentals_analyst:'基本面分析',bull_researcher:'多方研究员',bear_researcher:'空方研究员',
research_manager:'研究经理',trader:'交易员',aggressive_risk_analyst:'激进风险',
neutral_risk_analyst:'中性风险',conservative_risk_analyst:'保守风险',portfolio_manager:'组合经理'};
const order=Object.keys(roleLabels);let selected=null;let selectedManually=false;let workflowKey=null;
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=String(text);
if(cls)node.className=cls;return node}
function fillList(root,title,items){if(!items||!items.length)return;root.append(el('b',title));const list=el('ul',undefined,'list');
items.forEach(item=>list.append(el('li',item)));root.append(list)}
function showDetail(role,manual=false){selected=role;if(manual)selectedManually=true;const workflow=window.currentWorkflow;if(!workflow)return;
const item=workflow.roles[role];const root=document.getElementById('detail');root.replaceChildren();
document.querySelectorAll('.role').forEach(button=>button.classList.toggle('selected',button.dataset.role===role));
root.append(el('h2',roleLabels[role]));const chips=el('div',undefined,'chips');chips.append(el('span',item.status,'chip'));
if(item.report){chips.append(el('span',item.report.stance,'chip'));chips.append(el('span',Math.round(item.report.confidence*100)+'% 信心','chip'))}
root.append(chips);if(!item.report){root.append(el('p',item.status==='running'?'正在生成结构化结论...':'尚无报告','empty'));return}
root.append(el('p',item.report.summary));fillList(root,'依据',item.report.evidence);fillList(root,'风险',item.report.risks);
fillList(root,'输入异常',item.report.input_anomalies)}
function decision(rootId,value){const root=document.getElementById(rootId);root.replaceChildren();if(!value){root.append(el('p','等待中','empty'));return}
root.append(el('h3',(value.action||'')+' · 权重倍数 '+Number(value.weight_multiplier).toFixed(2)));
root.append(el('p',value.thesis));fillList(root,'风险',value.risks)}
function metric(root,label,value){const row=el('p');row.append(el('b',label+'：'));row.append(document.createTextNode(String(value??'-')));root.append(row)}
function renderFinMem(root,payload){const memory=payload.memory||{};root.append(el('h2','FinMem 三层记忆'));
['short','mid','long'].forEach(lane=>{const box=el('section',undefined,'panel');box.append(el('b',lane+' memory'));fillList(box,'记录',memory[lane]||[]);root.append(box)});
const decision=payload.decision||{};metric(root,'最新动作',decision.action);fillList(root,'引用证据',decision.memory_ids||[])}
function renderQuanta(root,payload){root.append(el('h2','QuantaAlpha 候选因子'));(payload.candidates||[]).forEach(item=>{const card=el('section',undefined,'panel');
metric(card,'Candidate',item.expression);metric(card,'Parent',item.parent);metric(card,'Gate rejection',item.rejection_reason);root.append(card)});
fillList(root,'Parent edges',(payload.edges||[]).map(edge=>(edge.parent||'')+' → '+(edge.child||'')));metric(root,'Champion',payload.champion&&payload.champion.expression)}
function renderArena(root,payload){root.append(el('h2','Alpha Arena Leaderboard'));(payload.leaderboard||[]).forEach(item=>{const row=el('section',undefined,'panel');
metric(row,'排名',item.rank);metric(row,'参赛者',item.name);metric(row,'状态',item.status);metric(row,'收益',item.total_return);metric(row,'回撤',item.max_drawdown);metric(row,'成本',item.costs);root.append(row)});
Object.entries(payload.equity||{}).forEach(([name,points])=>fillList(root,name+' equity',Object.entries(points).map(([day,value])=>day+' · '+value)))}
function renderExperiment(data){const experiment=data.experiment;if(!experiment)return;document.getElementById('experiment-panel').hidden=false;
document.querySelectorAll('.grid,.decision').forEach(node=>node.hidden=true);document.getElementById('title').textContent=experiment.kind+' · '+experiment.run_id;
document.getElementById('meta').textContent=experiment.provider+' · stage '+experiment.stage;document.getElementById('status').textContent=experiment.status;
const summary=document.getElementById('experiment-summary');summary.textContent='调用 '+String((experiment.payload||{}).calls||0)+' · '+experiment.stage;
const root=document.getElementById('experiment-detail');root.replaceChildren();const payload=experiment.payload||{};
if(experiment.kind==='finmem')renderFinMem(root,payload);else if(experiment.kind==='quanta-alpha')renderQuanta(root,payload);else if(experiment.kind==='alpha-arena')renderArena(root,payload)}
function render(data){if(data.mode==='experiment'){renderExperiment(data);return}const workflow=data.workflow;document.getElementById('status').textContent=data.command_status+' · 已完成 '+data.workflow_count+' 个工作流';
if(!workflow){if(data.reason)document.getElementById('activity').textContent=data.reason;return}const nextKey=workflow.ticker+'|'+workflow.as_of+'|'+data.workflow_count;if(nextKey!==workflowKey){workflowKey=nextKey;selected=null;selectedManually=false}window.currentWorkflow=workflow;document.getElementById('title').textContent=workflow.ticker+' · '+workflow.as_of;
document.getElementById('meta').textContent=workflow.provider+' · '+workflow.status;const root=document.getElementById('roles');root.replaceChildren();
let done=0;order.forEach(role=>{const item=workflow.roles[role];const button=el('button',undefined,'role '+item.status);button.dataset.role=role;
button.append(el('div',roleLabels[role],'role-name'));let note=item.status;if(item.report)note=item.report.stance+' · '+Math.round(item.report.confidence*100)+'%';
button.append(el('div',note,'role-note'));button.addEventListener('click',()=>showDetail(role,true));root.append(button);if(['completed','skipped','failed'].includes(item.status))done++});
document.getElementById('progress').style.width=(done/order.length*100)+'%';const active=workflow.active_role;
if(!selectedManually&&active)showDetail(active);else if(!selected)showDetail(active||order[0]);document.getElementById('activity').textContent=active?'正在运行：'+roleLabels[active]:workflow.status;
decision('proposal',workflow.proposal);decision('final',workflow.final_review)}
async function poll(){try{const response=await fetch('state',{cache:'no-store'});if(response.ok)render(await response.json())}catch(error){
document.getElementById('activity').textContent='命令已结束或 Dashboard 已断开；保留最后状态。'}finally{setTimeout(poll,500)}}poll();
</script></body></html>"""
