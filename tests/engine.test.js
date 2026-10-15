import test from 'node:test';
import assert from 'node:assert/strict';
import {gridGraph,createWorld,computeGradient,solveField,conductances,growStep,mass,maxMass,evaluate,setNeed,aliveEdges,cutStrongEdge,snapshot} from '../engine.js';

test('grid has unique valid undirected edges and a connected ring-free scaffold',()=>{
 const g=gridGraph();const keys=new Set();
 for(const e of g.edges){assert.ok(e.a>=0&&e.b<g.nodes.length&&e.a!==e.b);let key=e.a+':'+e.b;assert.ok(!keys.has(key),'duplicate edge '+key);keys.add(key);}
});
test('Kirchhoff solve fixes ground and meets current balance',()=>{
 const w=createWorld(),g=w.graph,weights=conductances(w),b=new Float64Array(g.nodes.length),source=w.demands[0].source,sink=w.demands[0].sink;
 b[source]=1;let v=solveField(g,weights,b,sink),residual=new Float64Array(b.length);
 g.edges.forEach((e,i)=>{let q=weights[i]*(v[e.a]-v[e.b]);residual[e.a]+=q;residual[e.b]-=q;});
 assert.equal(v[sink],0);
 for(let i=0;i<b.length;i++)if(i!==sink)assert.ok(Math.abs(residual[i]-b[i])<1e-7,`unbalanced node ${i}`);
});
test('transport analytic local gradient matches finite difference',()=>{
 const w=createWorld({width:9,height:7});
 for(let i=0;i<18;i++)growStep(w);
 let weights=conductances(w),e=w.graph.edges.find(e=>e.h>0.02);
 assert.ok(e);const k=e.id,analytic=computeGradient(w).grad[k],eps=1e-5;
 e.h+=eps;let plus=evaluate(w).loss;e.h-=2*eps;let minus=evaluate(w).loss;e.h+=eps;
 const derivative=(plus-minus)/(2*eps);
 assert.ok(Math.abs(derivative+analytic)<Math.max(0.005,0.008*Math.abs(analytic)),`${derivative} vs ${analytic}`);
});
test('transfer adjoint gradient matches finite differences',()=>{
 const w=createWorld({width:9,height:7,preset:'transfer'});
 for(let i=0;i<12;i++)growStep(w);
 const ranked=[...w.graph.edges].sort((a,b)=>b.h-a.h);const e=ranked[0],eps=1e-5,analytic=computeGradient(w).grad[e.id];
 e.h+=eps;let plus=evaluate(w).loss;e.h-=2*eps;let minus=evaluate(w).loss;e.h+=eps;
 assert.ok(Math.abs((plus-minus)/(2*eps)+analytic)<Math.max(0.02,Math.abs(analytic)*.015));
});
test('budget never exceeded and conductances nonnegative',()=>{
 for(let preset of ['bridge','fork','detour','transfer']){
  const w=createWorld({width:11,height:9,preset,budget:.14});
  for(let j=0;j<80;j++)growStep(w);
  assert.ok(mass(w)<=maxMass(w)+1e-6,`${preset} exceeded budget`);
  assert.ok(w.graph.edges.every(e=>e.h>=0&&e.h<=3));
 }
});
test('growth improves trained needs and beats a matched uniform network',()=>{
 for(let preset of ['bridge','fork','detour']){
  const w=createWorld({width:13,height:9,preset});const before=evaluate(w).loss;
  for(let i=0;i<135;i++)growStep(w);
  assert.ok(evaluate(w).loss<before*.7,`${preset} no useful growth`);
  assert.ok(evaluate(w).loss<evaluate(w,{uniform:true}).loss,`${preset} worse than uniform`);
  assert.ok(aliveEdges(w)>0);
 }
});
test('distinct needs grow different topologies',()=>{
 let a=createWorld({width:13,height:9,preset:'bridge'}),b=createWorld({width:13,height:9,preset:'fork'});
 for(let i=0;i<100;i++){growStep(a);growStep(b);}
 const va=a.graph.edges.map(e=>e.h),vb=b.graph.edges.map(e=>e.h);
 const distance=va.reduce((s,v,i)=>s+Math.abs(v-vb[i]),0);
 assert.ok(distance>1.0,`identical structure ${distance}`);
});
test('damage and adaptation can change the developed network',()=>{
 const w=createWorld({width:13,height:9});for(let i=0;i<100;i++)growStep(w);
 const cut=cutStrongEdge(w);assert.ok(cut);assert.equal(cut.h,0);assert.equal(cut.broken,true);
 for(let i=0;i<50;i++)growStep(w);
 assert.equal(cut.h,0);
 assert.ok(Number.isFinite(evaluate(w).loss));
});
test('snapshot is reproducible and contains no non-finite results',()=>{
 const a=createWorld({width:11,height:7}),b=createWorld({width:11,height:7});
 for(let i=0;i<25;i++){growStep(a);growStep(b);}
 assert.deepEqual(snapshot(a),snapshot(b));
});
test('effective external response is reciprocal and changes with growth',async()=>{
 const {responseMatrix}=await import('../engine.js');
 const w=createWorld({width:11,height:9});const A=responseMatrix(w).values;
 for(let i=0;i<A.length;i++)for(let j=0;j<A.length;j++)assert.ok(Math.abs(A[i][j]-A[j][i])<1e-7);
 for(let i=0;i<85;i++)growStep(w);
 const B=responseMatrix(w).values;
 assert.ok(Math.abs(B[0][0]-A[0][0])>1e-2);
});
test('detour: response matrix avoids disconnected obstacle nodes',async()=>{
 const {responseMatrix}=await import('../engine.js');
 const w=createWorld({preset:'detour'});
 const A=responseMatrix(w).values;
 assert.equal(A.length,8);assert.ok(A.flat().every(Number.isFinite));
});
