/* GrowingMatrix -- deterministic need-driven conductive network laboratory.
 * Passive Kirchhoff field: L(g) v = b, grounded at node r.
 * Routing compliance gradient: -(v_a-v_b)^2.
 * Supervised loss gradient: -(v_a-v_b)(z_a-z_b), Lz=C^T(Cv-target).
 * Adaptive conductances are projected onto a weighted material budget.
 * Dormant edges have a small fixed scaffold conductance: this is *not* node birth.
 */
export const VERSION = '1.0.0';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const hypot=Math.hypot;
const square=x=>x*x;
export function seedRandom(seed=1){let s=(seed>>>0)||1;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)/4294967296;};}
export function gridGraph(w=17,h=11,{detour=false}={}){
  if(w<7||h<7)throw new Error('Grid must be at least 7x7');
  const nodes=[];for(let y=0;y<h;y++)for(let x=0;x<w;x++)nodes.push({id:y*w+x,x,y});
  const edges=[];
  function add(a,b){
    const pa=nodes[a],pb=nodes[b],mx=(pa.x+pb.x)/2/(w-1),my=(pa.y+pb.y)/2/(h-1);
    // Remove edges through a centered block. Endpoints are still present as latent nodes.
    if(detour&&mx>.385&&mx<.615&&my>.27&&my<.73)return;
    const len=hypot(pa.x-pb.x,pa.y-pb.y);
    edges.push({id:edges.length,a,b,len,h:0,broken:false});
  }
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    let a=y*w+x;
    if(x+1<w)add(a,a+1);
    if(y+1<h)add(a,a+w);
    if(x+1<w&&y+1<h&&(x+y)%2===0)add(a,a+w+1); // one diagonal per cell
    // The alternate diagonal starts at top-right and ends at bottom-left.
    if(x+1<w&&y+1<h && (x+y)%2===1)add(a+1,a+w);
  }
  return {w,h,nodes,edges};
}
export const idAt=(g,x,y)=>clamp(Math.round(y),0,g.h-1)*g.w+clamp(Math.round(x),0,g.w-1);
function pin(g,x,y){return idAt(g,Math.round((g.w-1)*x),Math.round((g.h-1)*y));}
function source(g,x,y){return {node:pin(g,x,y),current:1};}
function route(g,a,b,weight=1){return {source:a,sink:b,weight};}
function makeTeacher(world){
  const g=world.graph,edges=g.edges;const wts=new Float64Array(edges.length);
  // A nonuniform but still passive target system, biased toward two channels.
  for(let i=0;i<edges.length;i++){
    let e=edges[i],a=g.nodes[e.a],b=g.nodes[e.b];
    let x=(a.x+b.x)/2/(g.w-1),y=(a.y+b.y)/2/(g.h-1);
    const corridor1=Math.exp(-square((y-(.22+.48*x))/.14));
    const corridor2=Math.exp(-square((y-(.78-.48*x))/.14));
    wts[i]=world.scaffold+.85*corridor1+.55*corridor2;
  }
  return wts;
}
export function setNeed(world,preset,{keepGrowth=false}={}){
  if(!['bridge','fork','detour','transfer'].includes(preset))throw new Error('Unknown need');
  if(world.preset!==preset && ((world.preset==='detour') !== (preset==='detour'))){
    const next=gridGraph(world.graph.w,world.graph.h,{detour:preset==='detour'});
    // Same geometry: carry learned conductances across shared edges, when requested.
    if(keepGrowth){let keys=new Map(world.graph.edges.map(e=>[e.a+':'+e.b,e.h]));
      for(let e of next.edges)e.h=keys.get(e.a+':'+e.b)||0;
    }
    world.graph=next;
  }else if(!keepGrowth){for(let e of world.graph.edges){e.h=0;e.broken=false;}}
  const g=world.graph,mid=Math.round(g.h/2), left=x=>idAt(g,1,x),right=x=>idAt(g,g.w-2,x);
  const yUpper=Math.max(1,mid-3),yLower=Math.min(g.h-2,mid+3);
  if(preset==='bridge'){
    world.demands=[route(g,left(mid),right(mid))];
    world.holdout=[route(g,left(mid-1),right(mid-1)),route(g,left(mid+1),right(mid+1))];
  }else if(preset==='fork'){
    world.demands=[route(g,left(mid),right(yUpper)),route(g,left(mid),right(yLower))];
    world.holdout=[route(g,left(mid-1),right(yUpper)),route(g,left(mid+1),right(yLower))];
  }else if(preset==='detour'){
    world.demands=[route(g,left(mid),right(mid))];
    world.holdout=[route(g,left(mid-1),right(mid-1)),route(g,left(mid+1),right(mid+1))];
  }else{
    world.demands=[];world.holdout=[];
    world.transfer={ground:right(mid),inputs:[left(yUpper),left(yLower)],outputs:[right(yUpper),right(yLower)]};
    const teacher=makeTeacher(world);
    const inputs=[[1,0],[0,1],[1,.6],[.6,1]],tests=[[.3,.7],[.8,.25],[.6,.6],[.15,.9]];
    const create=amps=>{
      const b=new Float64Array(g.nodes.length);
      world.transfer.inputs.forEach((n,i)=>b[n]=amps[i]);
      const v=solveField(g,teacher,b,world.transfer.ground);
      return {b,ground:world.transfer.ground,outputs:world.transfer.outputs,target:world.transfer.outputs.map(n=>v[n]),amps};
    };
    world.demands=inputs.map(create);world.holdout=tests.map(create);
  }
  world.preset=preset;world.step=0;world.history=[];world.last=null;
  return world;
}
export function createWorld({width=17,height=11,preset='bridge',budget=.24,rate=.055,seed=7}={}){
  const world={graph:gridGraph(width,height,{detour:preset==='detour'}),preset:null,scaffold:.025,
    budget,rate,seed,step:0,demands:[],holdout:[],history:[],last:null};
  return setNeed(world,preset);
}
export function conductances(world){return Float64Array.from(world.graph.edges,e=>e.broken?world.scaffold*.001:world.scaffold+e.h);}
function laplace(g,weights,x,ground,out){
  out.fill(0);
  for(let k=0;k<g.edges.length;k++){
    const e=g.edges[k],v=weights[k]*(x[e.a]-x[e.b]);
    out[e.a]+=v;out[e.b]-=v;
  }
  out[ground]=x[ground];
  return out;
}
export function solveField(graph,weights,rhs,ground,{tol=1e-9,maxIter=900}={}){
  const n=graph.nodes.length;
  if(!Number.isInteger(ground)||ground<0||ground>=n)throw new Error('Invalid grounded node');
  if(rhs.length!==n||weights.length!==graph.edges.length)throw new Error('Invalid dimensions');
  const x=new Float64Array(n),r=Float64Array.from(rhs),d=new Float64Array(n),z=new Float64Array(n),p=new Float64Array(n),Ap=new Float64Array(n);
  for(let k=0;k<graph.edges.length;k++){
    const e=graph.edges[k],w=weights[k];if(w<0||!Number.isFinite(w))throw new Error('Bad conductance');
    d[e.a]+=w;d[e.b]+=w;
  }
  r[ground]=0;
  let norm0=0,rz=0;for(let i=0;i<n;i++){z[i]=r[i]/Math.max(d[i],1e-9);p[i]=z[i];rz+=r[i]*z[i];norm0+=r[i]*r[i];}
  if(norm0===0)return x;
  const target=tol*tol*norm0;
  for(let iter=0;iter<maxIter;iter++){
    laplace(graph,weights,p,ground,Ap);
    let pAp=0;for(let i=0;i<n;i++)pAp+=p[i]*Ap[i];
    if(pAp<=0||!Number.isFinite(pAp))throw new Error('Unstable or disconnected network');
    let alpha=rz/pAp,err=0;
    for(let i=0;i<n;i++){x[i]+=alpha*p[i];r[i]-=alpha*Ap[i];err+=r[i]*r[i];}
    if(err<=target){x[ground]=0;return x;}
    let rz2=0;for(let i=0;i<n;i++){z[i]=r[i]/Math.max(d[i],1e-9);rz2+=r[i]*z[i];}
    let beta=rz2/Math.max(rz,1e-90);
    for(let i=0;i<n;i++)p[i]=z[i]+beta*p[i];
    p[ground]=0;rz=rz2;
  }
  x[ground]=0;
  return x;
}
function makeRhs(g,d){
  const b=new Float64Array(g.nodes.length);b[d.source]=1;return b;
}
export function sample(world,d,weights=conductances(world)){
  const g=world.graph;
  if(d.outputs){
    const v=solveField(g,weights,d.b,d.ground);
    const residual=d.outputs.map((n,i)=>v[n]-d.target[i]);
    return {v,loss:.5*residual.reduce((s,e)=>s+e*e,0),residual};
  }
  const v=solveField(g,weights,makeRhs(g,d),d.sink);
  return {v,loss:v[d.source],residual:[]};
}
export function evaluate(world,{holdout=false,uniform=false}={}){
  const items=holdout?world.holdout:world.demands;
  let weights=conductances(world);
  if(uniform){
    const mass=world.graph.edges.reduce((s,e)=>s+e.h*e.len,0);
    const totalLen=world.graph.edges.reduce((s,e)=>s+(e.broken?0:e.len),0);
    const amount=totalLen?mass/totalLen:0;
    weights=Float64Array.from(world.graph.edges,e=>e.broken?world.scaffold*.001:world.scaffold+amount);
  }
  let loss=0;let field=null;
  for(const d of items){let s=sample(world,d,weights);loss+=s.loss;field=field||s.v;}
  return {loss:loss/Math.max(items.length,1),field};
}
export function mass(world){return world.graph.edges.reduce((s,e)=>s+e.h*e.len,0);}
export function maxMass(world){return world.budget*world.graph.edges.reduce((s,e)=>s+(e.broken?0:e.len),0);}
export function aliveEdges(world,threshold=.055){return world.graph.edges.filter(e=>!e.broken&&e.h>threshold).length;}
function projectBudget(world,z){
  const edges=world.graph.edges,n=edges.length,B=maxMass(world);
  let m=0;const cap=3;
  const out=Float64Array.from(z,(v,i)=>edges[i].broken?0:clamp(v,0,cap));
  for(let i=0;i<n;i++)m+=out[i]*edges[i].len;
  if(m<=B)return out;
  let lo=0,hi=cap;
  for(let k=0;k<45;k++){
    let mid=(lo+hi)/2,sum=0;
    for(let i=0;i<n;i++)sum+=edges[i].len*clamp(z[i]-mid*edges[i].len,0,cap)*(edges[i].broken?0:1);
    if(sum>B)lo=mid;else hi=mid;
  }
  for(let i=0;i<n;i++)out[i]=edges[i].broken?0:clamp(z[i]-hi*edges[i].len,0,cap);
  return out;
}
export function computeGradient(world){
  const g=world.graph,weights=conductances(world),grad=new Float64Array(g.edges.length);
  let loss=0,vLast=null;
  for(const d of world.demands){
    let b=d.outputs?d.b:makeRhs(g,d),ground=d.outputs?d.ground:d.sink;
    let v=solveField(g,weights,b,ground),z=null;
    if(d.outputs){
      const adjoint=new Float64Array(g.nodes.length);
      for(let j=0;j<d.outputs.length;j++){let err=v[d.outputs[j]]-d.target[j];adjoint[d.outputs[j]]+=err;loss+=.5*err*err;}
      z=solveField(g,weights,adjoint,ground);
    }else loss+=v[d.source];
    for(let k=0;k<g.edges.length;k++){
      let e=g.edges[k],dv=v[e.a]-v[e.b];
      grad[k]+=d.outputs?dv*(z[e.a]-z[e.b]):dv*dv;
    }
    vLast=v;
  }
  let factor=1/Math.max(1,world.demands.length);
  for(let k=0;k<grad.length;k++)grad[k]*=factor;
  return {grad,loss:loss*factor,field:vLast};
}
export function growStep(world){
  const {grad,loss,field}=computeGradient(world);
  const edges=world.graph.edges;
  let absMax=1e-12;
  for(let i=0;i<grad.length;i++)if(!edges[i].broken)absMax=Math.max(absMax,Math.abs(grad[i])/edges[i].len);
  // Normalized projected gradient ascent in negative objective derivative direction.
  const raw=Float64Array.from(edges,(e,i)=>e.broken?0:e.h+world.rate*(grad[i]/e.len)/absMax);
  const projected=projectBudget(world,raw);
  for(let i=0;i<edges.length;i++)edges[i].h=projected[i];
  world.step++;
  world.last={field,gradient:grad,loss};
  if(world.step%4===0||world.step===1){
    const train=evaluate(world).loss,test=evaluate(world,{holdout:true}).loss,base=evaluate(world,{uniform:true}).loss;
    world.history.push({step:world.step,train,test,baseline:base,material:mass(world),active:aliveEdges(world)});
    if(world.history.length>260)world.history.shift();
  }
  return {loss,field};
}
export function cutStrongEdge(world){
  const g=world.graph, candidates=g.edges.filter(e=>!e.broken&&e.h>.02&&g.nodes[e.a].x>g.w*.24&&g.nodes[e.a].x<g.w*.74);
  const ranked=(candidates.length?candidates:g.edges.filter(e=>!e.broken&&e.h>.02)).sort((a,b)=>b.h-a.h);
  if(!ranked.length)return null;
  let e=ranked[0];e.h=0;e.broken=true;return e;
}
export function resetGrowth(world){for(const e of world.graph.edges){e.h=0;e.broken=false;}world.step=0;world.last=null;world.history=[];}
export function changeTerminals(world,role,node){
  if(world.preset==='transfer')return false;
  const active=world.graph.edges.some(e=>!e.broken&&(e.a===node||e.b===node));
  if(!active)return false;
  for(const d of world.demands){if((role==='source'&&d.sink===node)||(role==='sink'&&d.source===node))return false;}
  for(const d of world.demands){if(role==='source')d.source=node;else d.sink=node;}
  for(const d of world.holdout){if(role==='source')d.source=node;else d.sink=node;}
  world.history=[];world.step=0;return true;
}
export function snapshot(world){return {
  version:VERSION,preset:world.preset,step:world.step,budget:world.budget,rate:world.rate,seed:world.seed,
  dimensions:[world.graph.w,world.graph.h],edges:world.graph.edges.map(e=>({a:e.a,b:e.b,h:+e.h.toFixed(6),broken:e.broken})),
  training_loss:evaluate(world).loss,holdout_loss:evaluate(world,{holdout:true}).loss,
  uniform_matched_loss:evaluate(world,{uniform:true}).loss,material:mass(world),capacity:maxMass(world)
};}

/* Symmetric 8-port Green/response operator, measured through the entire network.
 * This is a restricted inverse grounded Laplacian, not a trained neural matrix.
 */
export function responseMatrix(world){
 const g=world.graph,ground=idAt(g,g.w-2,Math.round((g.h-1)/2)),weights=conductances(world);
 const ports=[
  idAt(g,1,1),idAt(g,1,Math.round((g.h-1)/2)),idAt(g,1,g.h-2),
  idAt(g,g.w-2,1),idAt(g,g.w-2,g.h-2),
  idAt(g,Math.round((g.w-1)/2),1),idAt(g,Math.round((g.w-1)/2),g.h-2),
  idAt(g,Math.round((g.w-1)*.25),Math.round((g.h-1)/2))
 ];
 const data=[];
 for(let j=0;j<ports.length;j++){
  const b=new Float64Array(g.nodes.length);b[ports[j]]=1;
  const v=solveField(g,weights,b,ground);
  data.push(ports.map(n=>v[n]));
 }
 return {ports,ground,values:ports.map((_,i)=>ports.map((_,j)=>data[j][i]))};
}
