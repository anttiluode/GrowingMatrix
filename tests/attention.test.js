import test from 'node:test';
import assert from 'node:assert/strict';
import {createWorld,configureAttention,growStep,evaluate,computeGradient,mass,maxMass,attentionReadout} from '../engine.js';
import {attentionTarget,attentionLoss,rowKL,unrestrictedReference,singlePortDiagnostics} from '../attention.js';

function make(task='mirror',layout='two'){
 const w=createWorld({preset:'attention',width:13,height:11});configureAttention(w,task,layout);return w;
}
test('all attention targets are stochastic and distinct',()=>{
 const targets=['mirror','shift','duplicate'].map(task=>attentionTarget(task));
 for(const A of targets){assert.equal(A.length,8);for(const row of A){assert.ok(row.every(v=>v>0));assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-12);}}
 assert.notDeepEqual(targets[0],targets[1]);assert.notDeepEqual(targets[1],targets[2]);
});
test('query-key circuits conserve sink uptake for single and two ports',()=>{
 for(const layout of ['two','single']){
  const w=make('shift',layout),out=attentionReadout(w);
  for(const row of out.matrix){assert.ok(row.every(v=>v>=-1e-9));assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-7);}
  assert.equal(out.matrix.length,8);
 }
});
test('co-located port Green operator is reciprocal and peak at injection',()=>{
 const w=make(),p=singlePortDiagnostics(w);
 assert.ok(p.symmetry<1e-7,p.symmetry);assert.equal(p.rowsAway,0);assert.equal(p.maxViolation,0);
});
test('attention adjoint gradient agrees with central differences',()=>{
 const w=make('duplicate');for(let i=0;i<30;i++)growStep(w);
 const grad=computeGradient(w).grad;
 for(const k of [8,17,31,53,78]){
  const e=w.graph.edges[k],h=e.h,eps=1e-4;
  e.h=h+eps;const plus=evaluate(w).loss;
  e.h=h-eps;const minus=evaluate(w).loss;
  e.h=h;
  const numeric=(plus-minus)/(2*eps);
  assert.ok(Math.abs(numeric+grad[k])<1e-5,`edge ${k} numeric ${numeric} vs ${-grad[k]}`);
 }
});
test('attention growth stays in budget and lowers all three errors',()=>{
 for(const task of ['mirror','shift','duplicate']){
  const w=make(task),before=evaluate(w).loss;
  for(let i=0;i<160;i++)growStep(w);
  const after=evaluate(w).loss;
  assert.ok(after<before-.001,`${task}: ${before}->${after}`);
  assert.ok(mass(w)<=maxMass(w)+1e-6);
  assert.ok(evaluate(w,{uniform:true}).loss>after);
 }
});
test('two independently trained spatial layouts differ on shift',()=>{
 const t=make('shift','two'),s=make('shift','single');
 for(let k=0;k<220;k++){growStep(t);growStep(s);}
 assert.ok(evaluate(t).loss<evaluate(s).loss);
 assert.ok(Number.isFinite(rowKL(t.attention.target,attentionReadout(t).matrix)));
});
test('sink stress is physical robustness, not held-out token inputs',()=>{
 const w=make('mirror');for(let k=0;k<50;k++)growStep(w);
 const a=evaluate(w).loss,b=evaluate(w,{holdout:true}).loss;
 assert.ok(Number.isFinite(a)&&Number.isFinite(b));assert.ok(Math.abs(a-b)>1e-8);
});
test('unrestricted direct bipartite reference has zero limiting error, unmatched cost',()=>{
 for(const task of ['mirror','shift','duplicate']){
  const r=unrestrictedReference(task);assert.equal(r.kl,0);
  assert.ok(attentionLoss(r.matrix,attentionTarget(task))<1e-12);
 }
});