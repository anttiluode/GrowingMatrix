import {createWorld,growStep,evaluate,configureAttention,attentionReadout,mass} from './engine.js';
import {rowKL,singlePortDiagnostics} from './attention.js';
const steps=Number(process.argv[2]||240);
if(!(Number.isInteger(steps)&&steps>=1&&steps<=10000))throw Error('steps must be 1..10000');
const rows=[];
for(const task of ['mirror','shift','duplicate'])for(const layout of ['two','single']){
 const world=createWorld({preset:'attention'});configureAttention(world,task,layout);
 for(let step=0;step<steps;step++)growStep(world);
 const A=attentionReadout(world).matrix;
 rows.push({task,layout,steps,loss:evaluate(world).loss,kl:rowKL(world.attention.target,A),
  sink_stress_loss:evaluate(world,{holdout:true}).loss,
  uniform_same_material_loss:evaluate(world,{uniform:true}).loss,
  material:mass(world),single_port_symmetry_error:singlePortDiagnostics(world).symmetry});
}
console.log(JSON.stringify({protocol:'GrowingMatrix Gate 5: planar fixed attention targets',
 limits:['Eight fixed query/key ports; no token-conditioned QK computation','No node birth',
 'Two-port and single-port models train separately within identical material budget caps',
 'Direct bipartite reference reaches KL→0 only as an unmatched spatial/complexity upper bound',
 'Sink-stress is NOT a held-out token experiment'],rows},null,2));