import {rowKL,singlePortDiagnostics} from './attention.js';
import {createWorld,setNeed,growStep,evaluate,mass,maxMass,aliveEdges,cutStrongEdge,resetGrowth,changeTerminals,snapshot,responseMatrix,configureAttention,attentionReadout} from './engine.js';
const $=q=>document.querySelector(q),$$=q=>Array.from(document.querySelectorAll(q));
const canvas=$('#network'),ctx=canvas.getContext('2d');
const historyCanvas=$('#history'),hctx=historyCanvas.getContext('2d');
const matrixCanvas=$('#matrix'),mctx=matrixCanvas.getContext('2d');
const presets={
 bridge:['One path through a field','A source and destination generate current. Helpful edges compete for a limited material supply.','BRIDGE / ONE SOURCE → ONE SINK'],
 fork:['A network that learns to branch','Two different destination demands share the same infrastructure. A fork can save material.','FORK / TWO DEMAND PATTERNS'],
 detour:['A solution around an absence','A block of candidate material is missing. Useful paths must reorganize around it.','DETOUR / FORBIDDEN REGION'],
 transfer:['A network that learns a mapping','Inputs create potential fields. A second, adjoint field gives local edges credit for matching target outputs.','TRANSFER / CLAMPED-ERROR GRADIENT'],
 attention:['Can a planar network grow attention?','Query ports inject current. Key ports drain it. The lattice learns a fixed target attention matrix under a material budget.','ATTENTION / QUERY → KEY CURRENT SPLIT']
};
let world=createWorld(),running=true,speed=4,tool='inspect',frame=0,hovered=-1,selected=-1,lastStats=0,lastMatrix=0,operator=null,plot={w:600,h:480,x0:45,y0:45,dx:20,dy:20};
let toastTimer=null,benchmarkRunning=false;
function alertUser(message){const el=$('#toast');el.textContent=message;el.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('show'),3400);}
function format(v){if(!Number.isFinite(v))return '—';if(Math.abs(v)>=100)return v.toFixed(1);if(Math.abs(v)>=10)return v.toFixed(2);if(Math.abs(v)>=.01)return v.toFixed(3);return v.toExponential(2);}
function resetCanvas(c){const rect=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,rect.width),h=Math.max(1,rect.height);if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);}const cx=c.getContext('2d');cx.setTransform(dpr,0,0,dpr,0,0);return {cx,w,h};}
function setPreset(p){
 const carry=$('#preserve').checked;
 setNeed(world,p,{keepGrowth:carry});
 $$('.preset').forEach(b=>b.classList.toggle('active',b.dataset.preset===p));
 $('#attentionControls').hidden=p!=='attention';$('#matrixChoice').hidden=p!=='attention';$('#attentionMetrics').hidden=p!=='attention';$('#attentionBenchmark').hidden=p!=='attention';
 $('#matrixMode').value='current';$('#attentionTask').value='mirror';$('#attentionLayout').value='two';
 const desc=presets[p];$('#stageTitle').textContent=desc[0];$('#taskExplain').textContent=desc[1];$('#canvasTag').textContent=desc[2];
 $('#trainName').textContent=p==='attention'?'Mean squared attention error':p==='transfer'?'Mean squared output error':'Effective transport resistance';
 $('#testCaption').textContent=p==='attention'?'±20% sink conductance':'Unseen inputs';
 $$('.tool').forEach(b=>b.disabled=(p==='transfer'||p==='attention')&&b.dataset.tool!=='inspect');
 if(p==='transfer'||p==='attention')selectTool('inspect');
 operator=null;lastMatrix=0;lastStats=0;selected=-1;renderStats();renderMatrix();
 alertUser(carry?'New need assigned. Existing branches were retained.':'New need assigned. Growth starts from dormant connections.');
}
function selectTool(name){tool=name;$$('.tool').forEach(b=>b.classList.toggle('selected',b.dataset.tool===name));$('#hint').textContent=name==='inspect'?'Click any node to inspect its connections.':`Click a node to place the ${name}; the old structure stays.`;}
$$('.preset').forEach(b=>b.addEventListener('click',()=>setPreset(b.dataset.preset)));
$('.tool').forEach(b=>b.addEventListener('click',()=>selectTool(b.dataset.tool)));
function changeAttention(){
 if(world.preset!=='attention')return;
 configureAttention(world,$('#attentionTask').value,$('#attentionLayout').value,{keepGrowth:$('#preserve').checked});
 selected=-1;operator=null;renderStats();renderMatrix();alertUser('Attention target/layout updated; '+($('#preserve').checked?'material retained.':'material reset.'));
}
$('#attentionTask').addEventListener('change',changeAttention);
$('#attentionLayout').addEventListener('change',changeAttention);
$('#matrixMode').addEventListener('change',renderMatrix);
$('#budget').addEventListener('input',e=>{world.budget=Number(e.target.value)/100;$('#budgetValue').textContent=e.target.value+'%';});
$('#speed').addEventListener('input',e=>{speed=Number(e.target.value);$('#speedValue').textContent=e.target.value+'×';});
function setPlaying(v){running=v;$('#play').textContent=v?'Ⅱ   Pause growth':'▶   Resume growth';$('#liveLabel').textContent=v?'GROWING':'PAUSED';}
$('#play').addEventListener('click',()=>setPlaying(!running));
$('#steps').addEventListener('click',()=>{for(let k=0;k<40;k++)growStep(world);renderStats();renderMatrix();alertUser('Advanced by 40 growth steps.');});
$('#reset').addEventListener('click',()=>{resetGrowth(world);selected=-1;renderStats();renderMatrix();alertUser('Conductances and damage reset. The need stays the same.');});
$('#cut').addEventListener('click',()=>{const e=cutStrongEdge(world);if(e){renderStats();renderMatrix();alertUser(`Connection ${e.a}–${e.b} cut. Resume growth to adapt.`);}else alertUser('No developed connection available to cut yet.');});
$('#download').addEventListener('click',()=>{
 const text=JSON.stringify(snapshot(world),null,2),blob=new Blob([text],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=url;a.download=`growingmatrix-${world.preset}-step-${world.step}.json`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);alertUser('Downloaded growth parameters and the complete edge matrix.');
});
$('#attentionBenchmark').addEventListener('click',async()=>{
 if(benchmarkRunning)return;
 benchmarkRunning=true;const btn=$('#attentionBenchmark');btn.disabled=true;
 const out=$('#benchmarkResults');out.textContent='Comparing planar layouts; this may take several seconds.';
 const results=[];
 try{
  for(const task of ['mirror','shift','duplicate']){
   for(const layout of ['two','single']){
    await new Promise(resolve=>setTimeout(resolve,15));
    const trial=createWorld({preset:'attention',budget:world.budget,rate:world.rate});
    configureAttention(trial,task,layout);
    for(let step=0;step<240;step++)growStep(trial);
    const measured=attentionReadout(trial).matrix;
    results.push({task,layout,loss:evaluate(trial).loss,kl:rowKL(trial.attention.target,measured),stress:evaluate(trial,{holdout:true}).loss});
    out.textContent=`Attention benchmark ${results.length}/6 cases completed.`;
   }
  }
  const table=document.createElement('table'),head=document.createElement('thead');
  head.innerHTML='<tr><th>TARGET</th><th>LAYOUT</th><th>KL ↓</th></tr>';table.append(head);
  const body=document.createElement('tbody');
  for(const item of results){
   const tr=document.createElement('tr');
   for(const value of [item.task,item.layout,format(item.kl)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}
   body.append(tr);
  }
  table.append(body);out.replaceChildren(table);
  const p=document.createElement('p');
  p.textContent='Fixed teacher targets, 240 steps, equal planar material budget caps. Unrestricted direct query–key conductances approach KL=0 but are NOT spatially or material matched. KL is teacher → measured.';
  out.append(p);
 }catch(e){out.textContent='Benchmark failed: '+e.message;}
 finally{benchmarkRunning=false;btn.disabled=false;}
});
$('#benchmark').addEventListener('click',async()=>{
 if(benchmarkRunning)return;benchmarkRunning=true;const btn=$('#benchmark');btn.disabled=true;const out=$('#benchmarkResults');out.textContent='Benchmark running — local, deterministic, no server.';
 const items=[];
 try{
  for(let p of ['bridge','fork','detour','transfer']){
   await new Promise(resolve=>setTimeout(resolve,15));
   const w=createWorld({preset:p,budget:world.budget,rate:world.rate});
   for(let i=0;i<240;i++)growStep(w);
   const v=evaluate(w,{holdout:true}).loss;
   const heldoutUniform=evaluate(w,{holdout:true,uniform:true}).loss;
   items.push({preset:p,trained:v,uniform:heldoutUniform,ratio:v/Math.max(1e-12,heldoutUniform)});
   out.textContent=`Benchmark: ${items.length}/4 needs evaluated.`;
  }
  const table=document.createElement('table'),head=document.createElement('thead');head.innerHTML='<tr><th>NEED</th><th>GROWN</th><th>UNIFORM</th></tr>';table.append(head);
  const tbody=document.createElement('tbody');for(const item of items){const tr=document.createElement('tr');for(const val of [item.preset,format(item.trained),format(item.uniform)]){const td=document.createElement('td');td.textContent=val;tr.append(td);}tbody.append(tr);}table.append(tbody);
  out.replaceChildren(table);const p=document.createElement('p');p.textContent='Held-out objectives · equal material · 240 steps. Transfer loss is not directly comparable to routing resistance.';out.append(p);
 }catch(err){out.textContent='Benchmark error: '+err.message;}
 finally{benchmarkRunning=false;btn.disabled=false;}
});
function computeGeometry(w,h){
 const g=world.graph,padX=Math.min(54,w*.083),padY=Math.min(63,h*.13);
 const width=Math.max(30,w-2*padX),height=Math.max(30,h-2*padY);
 plot={w,h,x0:padX,y0:padY,dx:width/(g.w-1),dy:height/(g.h-1)};
}
function pt(n){const a=world.graph.nodes[n];return [plot.x0+a.x*plot.dx,plot.y0+a.y*plot.dy];}
function draw(){
 const {w,h}=resetCanvas(canvas);computeGeometry(w,h);ctx.clearRect(0,0,w,h);
 // Faint geometry underlay and axis labels.
 ctx.save();ctx.strokeStyle='rgba(112,160,169,.063)';ctx.lineWidth=.7;
 for(let y=0;y<world.graph.h;y++){let yy=plot.y0+y*plot.dy;ctx.beginPath();ctx.moveTo(plot.x0,yy);ctx.lineTo(plot.x0+(world.graph.w-1)*plot.dx,yy);ctx.stroke();}
 for(let x=0;x<world.graph.w;x++){let xx=plot.x0+x*plot.dx;ctx.beginPath();ctx.moveTo(xx,plot.y0);ctx.lineTo(xx,plot.y0+(world.graph.h-1)*plot.dy);ctx.stroke();}
 if(world.preset==='detour'){
  const x=plot.x0+plot.dx*(world.graph.w-1)*.385,y=plot.y0+plot.dy*(world.graph.h-1)*.27,ww=plot.dx*(world.graph.w-1)*.23,hh=plot.dy*(world.graph.h-1)*.46;
  ctx.fillStyle='rgba(250,115,104,.08)';ctx.strokeStyle='rgba(250,115,104,.55)';ctx.setLineDash([4,6]);ctx.lineWidth=1.2;ctx.fillRect(x,y,ww,hh);ctx.strokeRect(x,y,ww,hh);ctx.setLineDash([]);ctx.fillStyle='#cb8887';ctx.font='9px ui-monospace, monospace';ctx.textAlign='center';ctx.fillText('MISSING MEDIUM',x+ww/2,y+hh/2);
 }
 const field=world.last?.field;
 let lo=Infinity,hi=-Infinity;
 if(field)for(const v of field){lo=Math.min(lo,v);hi=Math.max(hi,v);}
 const g=world.graph;
 for(let k=0;k<g.edges.length;k++){
  const e=g.edges[k],a=pt(e.a),b=pt(e.b),conductance=e.h;
  if(e.broken){ctx.strokeStyle='rgba(253,108,105,.24)';ctx.lineWidth=1;ctx.setLineDash([2,4]);}
  else if(conductance<.022){ctx.strokeStyle='rgba(122,158,169,.105)';ctx.lineWidth=.9;}
  else{ctx.strokeStyle=`rgba(104,226,203,${Math.min(.95,.23+Math.sqrt(conductance)*.55)})`;ctx.lineWidth=Math.min(6.7,1.1+Math.sqrt(conductance)*3.9);ctx.shadowColor='rgba(79,216,187,.5)';ctx.shadowBlur=conductance>.18?8:0;}
  ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.shadowBlur=0;ctx.setLineDash([]);
  // Small moving markers encode instantaneous current magnitude (not causal speed).
  if(conductance>.1&&field&&k%3===0){
   const dv=field[e.a]-field[e.b],current=(world.scaffold+e.h)*dv;
   if(Math.abs(current)>.015){let f=((frame*.009*Math.min(5,1+Math.abs(current)*1.6)+k*.171)%1);
    if(current<0)f=1-f;
    const x=a[0]*(1-f)+b[0]*f,y=a[1]*(1-f)+b[1]*f;
    ctx.fillStyle='rgba(250,243,198,.75)';ctx.beginPath();ctx.arc(x,y,1.65,0,Math.PI*2);ctx.fill();
   }
  }
 }
 for(const n of g.nodes){
  const p=pt(n.id),isActive=selected===n.id||hovered===n.id;
  let shade='rgba(137,180,184,.33)';
  if(field&&hi>lo+1e-8){let z=(field[n.id]-lo)/(hi-lo);shade=`rgba(${Math.round(110+100*z)},${Math.round(178-40*z)},${Math.round(190-55*z)},.57)`;}
  ctx.beginPath();ctx.arc(p[0],p[1],isActive?4:2,0,Math.PI*2);ctx.fillStyle=shade;ctx.fill();
  if(isActive){ctx.beginPath();ctx.arc(...p,10,0,Math.PI*2);ctx.strokeStyle='#d3fff2';ctx.lineWidth=1;ctx.stroke();}
 }
 const sources=[],sinks=[];
 if(world.preset==='attention'){
  sources.push(...world.attention.queries);sinks.push(...world.attention.keys);
 }else if(world.preset==='transfer'){
  for(let node of world.transfer.inputs)sources.push(node);
  for(let node of world.transfer.outputs)sinks.push(node);
  sinks.push(world.transfer.ground);
 }else{for(let d of world.demands){sources.push(d.source);sinks.push(d.sink);}}
 for(const [kind,items] of [['source',[...new Set(sources)]],['sink',[...new Set(sinks)]]]){
  for(const n of items){let p=pt(n),color=kind==='source'?'#ffbd7b':'#c3b3ff';ctx.beginPath();ctx.arc(...p,12,0,Math.PI*2);ctx.strokeStyle=color;ctx.lineWidth=1.3;ctx.globalAlpha=.34;ctx.stroke();ctx.globalAlpha=1;
   ctx.beginPath();ctx.arc(...p,5.4,0,Math.PI*2);ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=15;ctx.fill();ctx.shadowBlur=0;
   ctx.fillStyle=color;ctx.font='bold 10px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(kind==='source'?'IN':'OUT',p[0],p[1]-18);
  }
 }
 ctx.restore();
}
function historyDraw(){
 const {cx,w,h}=resetCanvas(historyCanvas);cx.clearRect(0,0,w,h);
 const pad={l:13,r:10,t:12,b:20},W=w-pad.l-pad.r,H=h-pad.t-pad.b,data=world.history;
 cx.strokeStyle='#24454d';cx.lineWidth=1;
 for(let j=0;j<4;j++){let y=pad.t+H*j/3;cx.beginPath();cx.moveTo(pad.l,y);cx.lineTo(pad.l+W,y);cx.stroke();}
 if(!data.length){cx.fillStyle='#78939b';cx.font='11px system-ui';cx.textAlign='center';cx.fillText('Start growth to record the objective',w/2,h/2);return;}
 const hi=Math.max(...data.flatMap(d=>[d.train,d.baseline])),lo=Math.max(0.0000001,Math.min(...data.flatMap(d=>[d.train,d.baseline]))*.5);
 const logHi=Math.log(hi+1e-12),logLo=Math.log(lo+1e-12),span=Math.max(1e-9,logHi-logLo),maxStep=Math.max(1,data.at(-1).step);
 function plotLine(key,color){cx.beginPath();for(let i=0;i<data.length;i++){let d=data[i],x=pad.l+W*d.step/maxStep,y=pad.t+H*(1-(Math.log(Math.max(1e-12,d[key]))-logLo)/span);if(i===0)cx.moveTo(x,y);else cx.lineTo(x,y);}cx.strokeStyle=color;cx.lineWidth=2;cx.stroke();}
 plotLine('baseline','#d49b6d');plotLine('train','#6ce4cc');cx.fillStyle='#809ca3';cx.font='9px ui-monospace,monospace';cx.textAlign='left';cx.fillText('0',pad.l,h-4);cx.textAlign='right';cx.fillText(String(maxStep),w-12,h-4);
}
function renderMatrix(){
 const attention=world.preset==='attention';
 const view=attention?$('#matrixMode').value:'reciprocal';
 let A;
 if(attention&&view!=='reciprocal'){
  A=view==='teacher'?world.attention.target:view==='single'?singlePortDiagnostics(world).matrix:attentionReadout(world).matrix;
  operator={values:A};
 }else{
  operator=responseMatrix(world);A=operator.values;
 }
 if(attention){
  const measured=attentionReadout(world).matrix;
  const diagnostic=singlePortDiagnostics(world);
  $('#attentionMetrics').textContent=`Teacher→grown KL: ${format(rowKL(world.attention.target,measured))}\nCo-located Green symmetry error: ${diagnostic.symmetry.toExponential(1)}\nCo-located maximum-principle violations: ${diagnostic.rowsAway}/8 rows\nTarget: ${world.attention.task} · layout: ${world.attention.layout}`;
  $('#matrixCaption').textContent={
   current:'Current fractions leaving eight grounded key sinks. Rows are query injections; columns are key exits.',
   teacher:'Requested attention pattern, fixed for the entire growth run.',
   single:'Co-located query/key ports measured through the SAME grown lattice (not a separately optimized network).',
   reciprocal:'Original symmetric port-voltage Green response; not a current-split attention matrix.'
  }[view];
 }else $('#matrixCaption').textContent='Symmetric grounded Green/voltage response of the grown passive lattice.';
 const {cx,w,h}=resetCanvas(matrixCanvas);
 cx.clearRect(0,0,w,h);const n=A.length,cell=Math.min((w-42)/n,(h-26)/n),total=n*cell,x0=(w-total)/2,y0=9,max=Math.max(...A.flat()),min=Math.min(...A.flat());
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){
  const v=(A[i][j]-min)/Math.max(1e-10,max-min),r=Math.round(21+45*v),g=Math.round(64+158*v),b=Math.round(73+125*v);
  cx.fillStyle=`rgb(${r},${g},${b})`;cx.fillRect(x0+j*cell+1,y0+i*cell+1,cell-2,cell-2);
 }
 cx.fillStyle='#9cafb4';cx.font='9px ui-monospace,monospace';cx.textAlign='center';for(let i=0;i<n;i++){cx.fillText(String(i+1),x0+cell*(i+.5),y0+total+13);cx.fillText(String(i+1),x0-10,y0+cell*(i+.5)+3);}
}
function renderStats(){
 let tr=evaluate(world),ho=evaluate(world,{holdout:true});
 // Fair comparison: both structures face the same hold-out samples.
 const base=evaluate(world,{holdout:true,uniform:true});
 $('#trainValue').textContent=format(tr.loss);$('#testValue').textContent=format(ho.loss);$('#baselineValue').textContent=format(base.loss);
 const gain=(1-ho.loss/Math.max(base.loss,1e-12))*100;
 $('#advantage').textContent=Number.isFinite(gain)?(gain>=0?'+':'')+gain.toFixed(1)+'%':'—';
 $('#advantage').style.color=gain>=0?'#68dac6':'#f5a08f';
 $('#stepLabel').textContent='STEP '+String(world.step).padStart(4,'0');
 $('#activeValue').textContent=`${aliveEdges(world)} / ${world.graph.edges.length}`;
 $('#massValue').textContent=(100*mass(world)/Math.max(1e-9,maxMass(world))).toFixed(1)+'%';
 $('#budgetBar').style.width=Math.min(100,100*mass(world)/Math.max(1e-9,maxMass(world)))+'%';
 if(selected>=0){const e=world.graph.edges.filter(e=>e.a===selected||e.b===selected),used=e.filter(e=>e.h>.055).length;
  $('#activityText').textContent=`Node ${selected} has ${e.length} candidate connections, ${used} recruited. Total material ${mass(world).toFixed(1)} / ${maxMass(world).toFixed(1)}.`;
 }else $('#activityText').textContent=`Step ${world.step}: ${aliveEdges(world)} recruited edges, ${mass(world).toFixed(1)} material units. All computation runs locally.`;
 historyDraw();
}
function pointerNode(e){let rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,p=world.graph;
 const xx=Math.round((x-plot.x0)/plot.dx),yy=Math.round((y-plot.y0)/plot.dy);
 if(xx<0||yy<0||xx>=p.w||yy>=p.h)return -1;
 const px=plot.x0+xx*plot.dx,py=plot.y0+yy*plot.dy;
 if(Math.hypot(px-x,py-y)>Math.max(13,Math.min(plot.dx,plot.dy)*.5))return -1;
 return yy*p.w+xx;
}
canvas.addEventListener('pointermove',e=>hovered=pointerNode(e));canvas.addEventListener('pointerleave',()=>hovered=-1);
canvas.addEventListener('click',e=>{
 const n=pointerNode(e);if(n<0)return;
 if(tool==='inspect'){selected=n;renderStats();return;}
 if(world.preset==='transfer'||world.preset==='attention'){alertUser('Terminal editing is available in routing modes.');return;}
 if(!changeTerminals(world,tool,n)){alertUser('That node cannot be used (blocked, damaged, or already the opposite terminal).');return;}
 selected=n;renderStats();renderMatrix();alertUser(`New ${tool} set to node ${n}. Continuing from current material.`);
});
let lastGrow=0,lastDraw=0;
function tick(t){
 if(running&&!benchmarkRunning&&t-lastGrow>105){try{for(let i=0;i<(world.preset==='attention'?Math.min(speed,2):speed);i++)growStep(world);}catch(err){setPlaying(false);alertUser('Solver error: '+err.message);}lastGrow=t;}
 if(t-lastDraw>33){frame++;draw();lastDraw=t;}
 if(t-lastStats>360){renderStats();lastStats=t;}
 if(t-lastMatrix>3500){renderMatrix();lastMatrix=t;}
 requestAnimationFrame(tick);
}
// Recompute the view dimensions whenever the layout changes.
window.addEventListener('resize',()=>{draw();historyDraw();renderMatrix();});
setPreset('bridge');renderStats();renderMatrix();requestAnimationFrame(tick);
