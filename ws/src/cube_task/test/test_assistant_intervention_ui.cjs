/* Running-task intervention uses real editor listeners and a small DOM; no model calls. */
"use strict";
const fs=require("fs"),path=require("path"),vm=require("vm"),assert=require("assert/strict");
const staticRoot=path.join(__dirname,"..","cube_task","assistant","static");
const editorSource=fs.readFileSync(path.join(staticRoot,"editor.js"),"utf8");
const drawing=[];
const drawingContext=new Proxy({}, {get:(target,key)=>key in target?target[key]:(...args)=>drawing.push([key,...args]),set:(target,key,value)=>(target[key]=value,true)});

class Element{
  constructor(tag="div"){
    this.tagName=tag;this.children=[];this.dataset={};this.attributes={};this._text="";this.value="";
    this.className="";this.classes=new Set();this.listeners=new Map();this.selectors=new Map();
    this.classList={toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name),add:name=>this.classes.add(name)};
  }
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>child.textContent).join("");}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;this._text="";}
  setAttribute(name,value){this.attributes[name]=String(value);}
  addEventListener(type,callback){const list=this.listeners.get(type)||[];list.push(callback);this.listeners.set(type,list);}
  async dispatch(type,fields={}){
    const event={button:0,clientX:210,clientY:140,pointerId:1,key:"",target:this,preventDefault(){this.prevented=true;},...fields};
    await Promise.all((this.listeners.get(type)||[]).map(callback=>callback(event)));
    return event;
  }
  querySelector(selector){
    const found=this.children.find(child=>child.tagName===selector);if(found)return found;
    if(!this.selectors.has(selector))this.selectors.set(selector,new Element(selector));
    return this.selectors.get(selector);
  }
  getBoundingClientRect(){return {left:0,top:0,width:420,height:280};}
  getContext(){return drawingContext;}
  focus(){}
  setPointerCapture(id){this.captured=id;}
  hasPointerCapture(id){return this.captured===id;}
  releasePointerCapture(){this.captured=null;}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const copy=value=>JSON.parse(JSON.stringify(value));
const requests=[];
const ui={state:null,connected:true,pending:false,trail:[]};
let state={
  status:"ready",busy:false,needs_reset:false,
  scene:{scene_version:7,objects:[],placement:{
    bounds:{x:[.10,.50],y:[-.20,.20],r_min:.10,r_max:.62},
    table_bounds:{x:[.10,.50],y:[-.20,.20]},
    pick_polygon:[[.20,-.10],[.40,-.10],[.40,.10],[.20,.10]],point_tolerance:.015,margin:.002,
  }},
  simulation:{scene_version:7,held_object:null,zones:[],pick_polygon:[[.20,-.10],[.40,-.10],[.40,.10],[.20,.10]],
    objects:[{id:"truth-apple",label:"apple",center:[.30,0,.025],yaw:0,size:[.05,.05,.05]}],
    editor:{editable:true,catalog:[],max_objects:6}},
};
const context={
  document:{getElementById:get,createElement:tag=>new Element(tag),querySelectorAll:()=>[]},
  window:{devicePixelRatio:1,RobotGeometry:{primitives:()=>[],dimensions:object=>object.size||[.05,.05,.05],bounds:()=>({low:[-.025,-.025,0],high:[.025,.025,.05]})}},
  ResizeObserver:class{constructor(callback){this.callback=callback;}observe(){}},
  ui,$:get,palette:{red:"#cc7771",green:"#7cab90",yellow:"#d1b76d"},
  colorKey:value=>String(value||"").includes("red")?"red":null,
  displayLabel:value=>({apple:"苹果",pear:"梨",banana:"香蕉"})[value]||String(value),
  number:(value,fallback=0)=>Number.isFinite(Number(value))?Number(value):fallback,
  setText:(id,value)=>get(id).textContent=value,
  showError:message=>get("inline-error").textContent=message||"",
  updateControls:()=>{},
  renderState:value=>{state=value;ui.state=value;context.window.SceneEditor.update(value);},
  api:async(url,payload)=>{
    requests.push({url,payload:payload===undefined?undefined:copy(payload)});
    if(url==="/api/scene-edit"){
      state=copy(state);state.scene.scene_version++;state.simulation.scene_version++;
      if(payload.action==="clear")state.simulation.objects=[];
      return {ok:true};
    }
    if(url==="/api/scene-intervene")return {ok:true,queued:true,request_id:requests.filter(item=>item.url===url).length};
    return url==="/api/state"?copy(state):{ok:true};
  },
};
vm.createContext(context);
vm.runInContext(editorSource,context);
const editor=context.window.SceneEditor,map=get("editor-canvas");
let checks=0;
const eq=(actual,expected)=>{assert.equal(actual,expected);checks++;};
const deep=(actual,expected)=>{assert.deepEqual(copy(actual),copy(expected));checks++;};
const has=(text,pattern)=>{assert.match(text,pattern);checks++;};
const update=next=>{state=copy(next);ui.state=state;editor.update(state);};
(async()=>{
  state.busy=true;state.status="executing";state.events=[];
  state.simulation.editor={...state.simulation.editor,editable:false,intervention_available:true,intervention_pending:0};
  update(state);
  eq(get("editor-mode-edit").disabled,true);
  eq(get("editor-mode-place").disabled,true);
  eq(get("editor-mode-intervene").disabled,false);
  await get("editor-mode-intervene").dispatch("click");
  eq(get("editor-mode-intervene").attributes["aria-pressed"],"true");
  eq(map.attributes["aria-disabled"],"false");
  eq(get("editor-lock").hidden,true);
  eq(get("editor-delete").disabled,true);
  eq(get("editor-clear").disabled,true);
  has(get("editor-hint").textContent,/模拟现场变化/);
  await map.dispatch("pointerdown");
  eq(get("editor-yaw").disabled,false);
  // Polling while a task runs must not cancel a live intervention drag.
  update(state);
  await map.dispatch("pointermove",{clientX:235});
  await map.dispatch("pointerup",{clientX:235});
  await new Promise(resolve=>setImmediate(resolve));
  const intervention=copy(requests.find(item=>item.url==="/api/scene-intervene"));
  eq(intervention.payload.action,"move");
  eq(intervention.payload.object_id,"truth-apple");
  eq(intervention.payload.scene_version,7);
  eq(intervention.payload.xy[0]>.30,true);
  eq(intervention.payload.yaw,0);
  eq(ui.pending,false); // Stop and command controls do not share a scene-edit worker.
  eq(get("editor-mode-intervene").disabled,true);
  has(get("editor-lock").querySelector("span").textContent,/等待.*生效/);
  eq(requests.filter(item=>item.url==="/api/scene-edit").length,0);

  // A normal observation version update does not prove that the queued mutation happened.
  update({...state,simulation:{...state.simulation,scene_version:8}});
  eq(get("editor-mode-intervene").disabled,true);
  update({...state,events:[{kind:"scene_intervention_applied",request_id:1,message:"干预已生效，相机将确认变化"}]});
  eq(get("editor-mode-intervene").disabled,false);
  has(get("editor-hint").textContent,/干预已生效/);
  await map.dispatch("keydown",{key:"Delete"});
  await get("editor-clear").dispatch("click");
  eq(requests.filter(item=>item.url==="/api/scene-edit").length,0);
  eq(requests.filter(item=>item.url==="/api/scene-intervene").length,1);

  // Holding one object still permits interference with the other objects only.
  update({...state,simulation:{...state.simulation,held_object:"truth-apple",objects:state.simulation.objects.map(item=>({...item,held:true}))}});
  await map.dispatch("pointerdown");
  await map.dispatch("pointermove",{clientX:235});
  await map.dispatch("pointerup",{clientX:235});
  await get("editor-yaw").dispatch("change",{target:{value:45}});
  eq(requests.filter(item=>item.url==="/api/scene-intervene").length,1);
  eq(get("editor-yaw").disabled,true);
  has(get("editor-error").textContent,/夹持中的物品不能移动/);

  update({...state,simulation:{...state.simulation,held_object:null,objects:state.simulation.objects.map(item=>({...item,held:false}))}});
  await map.dispatch("pointerdown");
  await get("editor-yaw").dispatch("change",{target:{value:45}});
  await new Promise(resolve=>setImmediate(resolve));
  const rotation=requests.filter(item=>item.url==="/api/scene-intervene").at(-1);
  eq(rotation.payload.action,"move");
  eq(rotation.payload.yaw,Math.PI/4);
  update({...state,events:[...state.events,{kind:"scene_intervention_rejected",request_id:2,message:"目标被夹持，干预未生效"}]});
  eq(get("editor-mode-intervene").disabled,false);
  has(get("editor-error").textContent,/干预未生效/);
  ui.connected=false;editor.update(state);
  eq(get("editor-mode-intervene").disabled,true);
  ui.connected=true;
  update({...state,needs_reset:true});
  eq(get("editor-mode-intervene").disabled,true);
  eq(map.attributes["aria-disabled"],"true");
  console.log("Intervention UI real listeners: "+checks+" assertions passed.");
})().catch(error=>{console.error(error);process.exitCode=1;});
