/* Real UI listeners with a small DOM; no browser, API server or model calls. */
"use strict";
const fs=require("fs"),path=require("path"),vm=require("vm"),assert=require("assert/strict");
const staticRoot=path.join(__dirname,"..","cube_task","assistant","static");
const editorSource=fs.readFileSync(path.join(staticRoot,"editor.js"),"utf8");
const appSource=fs.readFileSync(path.join(staticRoot,"app.js"),"utf8");
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
  window:{devicePixelRatio:1,RobotGeometry:{primitives:()=>[],dimensions:object=>object.size||[.05,.05,.05]}},
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
    return url==="/api/state"?copy(state):{ok:true};
  },
};
vm.createContext(context);
vm.runInContext(editorSource,context);
vm.runInContext(appSource.slice(appSource.indexOf("async function submitInstruction("),appSource.indexOf("// Decode only received characters")),context);
vm.runInContext(appSource.slice(appSource.indexOf("function renderBatch("),appSource.indexOf("function renderGraspFeedback(")),context);
const editor=context.window.SceneEditor,map=get("editor-canvas");
let checks=0;
const eq=(actual,expected)=>{assert.equal(actual,expected);checks++;};
const deep=(actual,expected)=>{assert.deepEqual(copy(actual),copy(expected));checks++;};
const has=(text,pattern)=>{assert.match(text,pattern);checks++;};
const update=next=>{state=copy(next);ui.state=state;editor.update(state);};
const choose=async(fields={})=>{
  await get("editor-mode-place").dispatch("click");
  await map.dispatch("pointerdown",fields);
  await map.dispatch("pointerup",fields);
};

(async()=>{
  update(state);
  eq(editor.selection(),null);
  await choose();
  const selected=copy(editor.selection());
  eq(selected.scene_version,7);
  eq(selected.destination.type,"point");
  eq(Math.abs(selected.destination.xy[0]-.30)<1e-12,true);
  eq(Math.abs(selected.destination.xy[1])<1e-12,true);
  eq(requests.length,0); // Clicking the existing object's center still only selects a point.
  has(get("editor-destination-info").textContent,/0\.300/);
  has(get("editor-destination-info").textContent,/0\.000/);
  const latestDrawing=drawing.slice(drawing.findLastIndex(command=>command[0]==="clearRect"));
  eq(latestDrawing.some(command=>command[0]==="arc"&&Math.abs(command[1]-210)<1&&Math.abs(command[2]-140)<1),true);
  eq(latestDrawing.some(command=>command[0]==="fillText"&&/选定.*点/.test(command[1])),true);
  eq(get("editor-mode-place").attributes["aria-pressed"],"true");
  eq(get("editor-delete").disabled,true);
  eq(get("editor-yaw").disabled,true);
  const detached=editor.selection();detached.destination.xy[0]=.45;
  deep(editor.selection(),selected);
  await map.dispatch("pointermove",{clientX:225,clientY:150});
  await map.dispatch("pointerup",{clientX:225,clientY:150});
  eq(requests.length,0);

  get("instruction").value="把苹果放到选定位置";
  await context.submitInstruction("  把苹果放到选定位置  ");
  deep(requests.at(-1),{url:"/api/command",payload:{instruction:"把苹果放到选定位置",...selected}});
  eq(get("instruction").value,"");
  eq(ui.pending,false);

  await get("editor-destination-clear").dispatch("click");
  eq(editor.selection(),null);
  eq(requests.filter(request=>request.url==="/api/scene-edit").length,0);
  await choose({clientX:1000});
  eq(editor.selection(),null);
  has(get("editor-error").textContent,/范围/);
  await context.submitInstruction("把苹果放回抓取区");
  deep(requests.at(-1),{url:"/api/command",payload:{instruction:"把苹果放回抓取区"}});

  for(const locked of [{busy:true},{needs_reset:true}]){
    update({...state,...locked});
    const before=copy(editor.selection()),count=requests.length;
    await choose({clientX:225});
    deep(editor.selection(),before);
    eq(get("editor-mode-place").disabled,true);
    await context.submitInstruction("抓苹果");
    eq(requests.length,count);
    update({...state,busy:false,needs_reset:false});
  }
  ui.connected=false;editor.update(state);
  await choose({clientX:225});
  eq(editor.selection(),null);
  eq(get("editor-mode-place").disabled,true);
  ui.connected=true;editor.update(state);

  await choose();
  eq(editor.selection().scene_version,7);
  update({...state,scene:{...state.scene,scene_version:8}});
  eq(editor.selection(),null);
  has(get("editor-destination-info").textContent,/未选择/);
  update({...state,simulation:{...state.simulation,scene_version:8}});

  await choose();
  await get("reset-button").dispatch("click");
  eq(editor.selection(),null);
  eq(requests.at(-1).url,"/api/reset");
  update({...state,status:"ready",scene:{scene_version:9,objects:[],placement:copy(selectedPlacement())},simulation:{...state.simulation,scene_version:9}});

  await choose();
  await get("editor-mode-edit").dispatch("click");
  await get("editor-clear").dispatch("click");
  await new Promise(resolve=>setImmediate(resolve));
  eq(requests.filter(request=>request.url==="/api/scene-edit").length,1);
  eq(editor.selection(),null);

  context.renderBatch({batch:{status:"running",total:3,completed:0,current_index:1,items:[
    {index:1,label:"apple",zone_name:null,destination:{type:"region",name:"pick_area"},status:"executing"},
    {index:2,label:"pear",zone_name:null,destination:{type:"point",xy:[.3,.08]},status:"skipped"},
    {index:3,label:"banana",zone_name:"zone_yellow",status:"pending"},
  ]}});
  const rows=get("batch-items").children;
  eq(rows[0].children[1].children[0].textContent,"苹果 → 抓取区");
  has(rows[1].children[1].children[0].textContent,/梨 → 桌面点.*0\.300.*0\.080/);
  eq(rows[1].children[2].textContent,"已在目标位置");
  eq(rows[2].children[1].children[0].textContent,"香蕉 → 黄色区域");
  console.log("Placement UI real listeners: "+checks+" assertions passed.");
})().catch(error=>{console.error(error);process.exitCode=1;});

function selectedPlacement(){return {
  bounds:{x:[.10,.50],y:[-.20,.20],r_min:.10,r_max:.62},table_bounds:{x:[.10,.50],y:[-.20,.20]},
  pick_polygon:[[.20,-.10],[.40,-.10],[.40,.10],[.20,.10]],point_tolerance:.015,margin:.002,
};}
