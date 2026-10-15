/* GrowingMatrix Gate 5: fixed attention targets and physical current division.
 * No transformer or token-conditioned QK computation is performed here. */
import {idAt,conductances,solveShunted,mass} from './engine.js';
export const ATTENTION_SIZE=8;
export const ATTENTION_TASKS=['mirror','shift','duplicate'];
const EPS=1e-10;
export function attentionTarget(task,n=ATTENTION_SIZE){
 if(!ATTENTION_TASKS.includes(task))throw Error('Unknown attention target');
 const out=Array.from({length:n},()=>Array(n).fill(0));
 const tail=.025;
 for(let i=0;i<n;i++){
  for(let j=0;j<n;j++)out[i][j]=tail/n;
  if(task==='mirror')out[i][n-1-i]+=1-tail;
  if(task==='shift')out[i][(i+n-1)%n]+=1-tail;
  if(task==='duplicate'){
   const tokens=[1,2,1,3,4,5,3,6],peers=[];
   for(let j=0;j<n;j++)if(i!==j&&tokens[i]===tokens[j])peers.push(j);
   if(peers.length){out[i][i]+=.15*(1-tail);for(const j of peers)out[i][j]+=.85*(1-tail)/peers.length;}
   else for(let j=0;j<n;j++)out[i][j]+=(1-tail)/n;
  }
 }
 return out;
}
export function makeAttention(world,{task='mirror',layout='two'}={}){
 if(!ATTENTION_TASKS.includes(task)||!['two','single'].includes(layout))throw Error('Invalid attention configuration');
 const g=world.graph;
 if(g.h<10||g.w<9)throw Error('Attention requires a lattice >=9x10');
 const ys=Array.from({length:ATTENTION_SIZE},(_,i)=>1+Math.round(i*(g.h-3)/(ATTENTION_SIZE-1)));
 const left=ys.map(y=>idAt(g,1,y)),right=ys.map(y=>idAt(g,g.w-2,y));
 return {task,layout,queries:left,keys:layout==='two'?right:left.slice(),sinkConductance:3,
         target:attentionTarget(task)};
}
export function setAttention(world,task,layout=world.attention?.layout||'two',{keepGrowth=false}={}){
 if(!world.attention)throw Error('Attention not active');
 world.attention=makeAttention(world,{task,layout});
 if(!keepGrowth)for(const e of world.graph.edges){e.h=0;e.broken=false;}
 world.history=[];world.step=0;world.last=null;
}
function conductorSet(world,uniform){
 if(!uniform)return conductances(world);
 const totalLen=world.graph.edges.reduce((s,e)=>s+(e.broken?0:e.len),0);
 const equal=totalLen?mass(world)/totalLen:0;
 return Float64Array.from(world.graph.edges,e=>e.broken?world.scaffold*.001:world.scaffold+equal);
}
export function attentionReadout(world,{uniform=false,stress=false,layout=world.attention?.layout||'two'}={}){
 if(!world.attention)throw Error('Attention not active');
 const a=world.attention,g=world.graph,n=ATTENTION_SIZE,weights=conductorSet(world,uniform);
 const queries=a.queries,keys=layout==='two'?makeAttention(world,{task:a.task,layout}).keys:a.queries;
 const shunts=new Float64Array(g.nodes.length),currentSink=new Float64Array(n);
 for(let j=0;j<n;j++){
  const factor=stress?(j%2===0?0.8:1.2):1;
  currentSink[j]=a.sinkConductance*factor;
  shunts[keys[j]]+=currentSink[j];
 }
 const result=[],potentials=[];
 for(let i=0;i<n;i++){
  const rhs=new Float64Array(g.nodes.length);rhs[queries[i]]=1;
  const v=solveShunted(g,weights,rhs,shunts);
  potentials.push(v);
  result.push(keys.map((node,j)=>currentSink[j]*v[node]));
 }
 return {matrix:result,potentials,shunts,weights,keys,queries};
}
export function attentionLoss(A,T){
 let loss=0,n=T.length;
 for(let i=0;i<n;i++)for(let j=0;j<n;j++)loss+=.5*(A[i][j]-T[i][j])**2/n;
 return loss;
}
export function rowKL(T,A){
 let kl=0,n=T.length;
 for(let i=0;i<n;i++)for(let j=0;j<n;j++)kl+=T[i][j]*Math.log((T[i][j]+EPS)/(A[i][j]+EPS))/n;
 return kl;
}
export function evaluateAttention(world,{holdout=false,uniform=false}={}){
 const out=attentionReadout(world,{stress:holdout,uniform});
 return {loss:attentionLoss(out.matrix,world.attention.target),kl:rowKL(world.attention.target,out.matrix),field:out.potentials[0]};
}
export function gradientAttention(world){
 const {matrix,potentials,weights,shunts,keys}=attentionReadout(world);
 const graph=world.graph,a=world.attention,n=ATTENTION_SIZE,grad=new Float64Array(graph.edges.length);
 for(let i=0;i<n;i++){
  const v=potentials[i],rhs=new Float64Array(graph.nodes.length);
  for(let j=0;j<n;j++)rhs[keys[j]]+=a.sinkConductance*(matrix[i][j]-a.target[i][j])/n;
  const z=solveShunted(graph,weights,rhs,shunts);
  for(let k=0;k<graph.edges.length;k++){
   const e=graph.edges[k];grad[k]+=(v[e.a]-v[e.b])*(z[e.a]-z[e.b]);
  }
 }
 return {grad,loss:attentionLoss(matrix,a.target),field:potentials[0]};
}
export function unrestrictedReference(task,g=1e7){
 const target=attentionTarget(task);
 return {matrix:target,loss:0,kl:0,conductances:target.map(r=>r.slice()),groundLimit:g};
}
export function singlePortDiagnostics(world){
 const o=attentionReadout(world,{layout:'single'}),A=o.matrix,n=A.length;
 const C=o.potentials.map(v=>o.queries.map(key=>v[key]));
 let symmetry=0,maxViolation=0,rowsAway=0;
 for(let i=0;i<n;i++){
  for(let j=0;j<n;j++)symmetry=Math.max(symmetry,Math.abs(C[i][j]-C[j][i]));
  const peak=Math.max(...C[i]);
  if(peak>C[i][i]+1e-7){rowsAway++;maxViolation=Math.max(maxViolation,peak-C[i][i]);}
 }
 return {symmetry,maxViolation,rowsAway,matrix:A};
}
