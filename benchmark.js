import {createWorld,growStep,evaluate,mass,aliveEdges,snapshot,cutStrongEdge} from './engine.js';
const result=[];
for(const preset of ['bridge','fork','detour','transfer']){
  const world=createWorld({preset});
  let before=evaluate(world).loss;
  for(let i=0;i<240;i++)growStep(world);
  let train=evaluate(world).loss,hold=evaluate(world,{holdout:true}).loss;
  let base=evaluate(world,{holdout:true,uniform:true}).loss;
  let n=aliveEdges(world),material=mass(world);
  result.push({preset,initial:before,train,heldout:hold,heldoutUniform:base,activeEdges:n,material});
}
console.log('Need-driven growth benchmark (240 iterations each, deterministic)');
console.table(result.map(x=>Object.fromEntries(Object.entries(x).map(([k,v])=>[k,typeof v==='number'?+v.toFixed(5):v]))));
console.log('Note: baseline is a uniform network at the SAME material used by each grown network.');
