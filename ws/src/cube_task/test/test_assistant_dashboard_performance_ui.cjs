/* Exercise real polling/render paths without a browser, robot, or model. */
"use strict";
const fs=require("fs"),path=require("path"),vm=require("vm"),assert=require("assert/strict");
const source=fs.readFileSync(path.join(__dirname,"..","cube_task","assistant","static","app.js"),"utf8");
class Element{
  constructor(){this.textContent="";this.dataset={};this.hidden=false;this.listeners=new Map();this.loads=0;this.classList={toggle(){}};}
  replaceChildren(){}
  set src(value){this.url=value;this.loads++;}
  addEventListener(kind,callback){this.listeners.set(kind,callback);}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const copy=value=>JSON.parse(JSON.stringify(value));
const ui={connected:true,trail:[],cameraMode:"live",cameraViews:[],lastCameraAt:0,cameraLoading:false};
const view={dirty:false},requests=[];
let now=10000;
const context={ui,view,$:get,window:{},document:{createElement:()=>new Element()},Date:{now:()=>now},
  number:(value,fallback=0)=>Number.isFinite(Number(value))?Number(value):fallback,
  vec:value=>value.slice(),sub:(a,b)=>a.map((v,i)=>v-b[i]),length:a=>Math.hypot(...a),
  setText:(id,value)=>get(id).textContent=String(value),setConnection(){},updateControls(){},showError(){},
  renderMessages(){},renderPipeline(){},renderObservations(){},stateLabels:{ready:"场景就绪"},
  setTimeout(){},api:async url=>{requests.push(url);return copy(ui.state);}};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf("function refreshCamera("),source.indexOf("function selectedId(")),context);
vm.runInContext(source.slice(source.indexOf("function selectedId("),source.indexOf("// Dependency-free orthographic orbit renderer")),context);
const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const initial={status:"ready",busy:false,scene:{objects:[],class_to_zone:{}},simulation:{sim_time:1,
  objects:[{id:"apple",label:"apple",center:[.2,0,.03],transform:copy(identity),size:[.05,.05,.05]}],
  tcp_xyz:[.2,0,.2],tcp_transform:copy(identity),link_frames:[copy(identity)],
  gripper_width:.05,zones:[],table_z:0,physics:{physics_steps:1}}};
context.renderState(copy(initial));assert.equal(view.dirty,true);view.dirty=false;
// Time, feedback and status stay fresh without rebuilding identical geometry.
const progress=copy(initial);progress.simulation.sim_time=2;progress.simulation.physics.physics_steps=50;
progress.simulation.feedback={checks:[{ok:false,message:"仍需检查"}]};progress.status="executing";
context.renderState(progress);assert.equal(view.dirty,false);assert.equal(ui.state,progress);
assert.equal(get("sim-time").textContent,"2.0");
// Every visual input invalidates, including edits without advancing the clock.
const changes=[
  state=>state.simulation.objects[0].transform[0][3]+=.000001,
  state=>state.simulation.tcp_transform[1][3]+=.01,
  state=>state.simulation.tcp_xyz[0]+=.01,
  state=>state.simulation.link_frames[0][2][3]+=.01,
  state=>state.simulation.gripper_width=.04,
  state=>state.simulation.table_z=.01,
  state=>state.simulation.table_bounds={min_x:-.2},
  state=>state.simulation.zones.push({name:"zone_red",center:[.3,0,0],size:[.2,.2]}),
  state=>state.selected_object_id="apple",
  state=>state.simulation.objects[0].held=true,
];
for(const change of changes){context.renderState(copy(initial));view.dirty=false;const state=copy(initial);change(state);context.renderState(state);assert.equal(view.dirty,true);}
// Repaint requests from orbit/resize remain latched across unchanged polling.
context.renderState(copy(initial));view.dirty=true;context.renderState(copy(initial));assert.equal(view.dirty,true);

const camera=get("camera-image");ui.state={camera:{mode:"live",available:true,updated_at:123}};
context.refreshCamera();assert.equal(camera.loads,1);camera.listeners.get("load")();
now+=400;context.refreshCamera();assert.equal(camera.loads,1);
ui.state.camera.updated_at=124;context.refreshCamera();assert.equal(camera.loads,2);
// A failed image can retry the same frame; deduplication must not freeze it.
camera.listeners.get("error")();now+=400;context.refreshCamera();assert.equal(camera.loads,3);
camera.listeners.get("load")();ui.state.camera.updated_at=null;now+=400;context.refreshCamera();assert.equal(camera.loads,4);
camera.listeners.get("load")();now+=400;context.refreshCamera();assert.equal(camera.loads,5);
// Frozen views still refetch after a scene revision, even at the same stamp.
camera.listeners.get("load")();ui.cameraMode="observation:1";ui.cameraViews=[{id:"1",sim_time:1}];ui.state.scene={scene_version:1};
context.refreshCamera();assert.equal(camera.loads,6);camera.listeners.get("load")();
context.refreshCamera();assert.equal(camera.loads,6);ui.state.scene.scene_version=2;context.refreshCamera();assert.equal(camera.loads,7);

(async()=>{await context.poll();assert.equal(requests.at(-1),"/api/state?view=dashboard");console.log("Dashboard visual invalidation, frame deduplication and polling checks passed.");})().catch(error=>{console.error(error);process.exitCode=1;});
