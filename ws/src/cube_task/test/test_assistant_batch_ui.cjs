/* Standalone DOM simulation; no browser, server, credentials or model calls. */
"use strict";
const fs=require("fs"),path=require("path"),vm=require("vm"),assert=require("assert/strict");
const staticRoot=path.join(__dirname,"..","cube_task","assistant","static");
const source=fs.readFileSync(path.join(staticRoot,"app.js"),"utf8");

class Element{
  constructor(tag="div"){
    this.tagName=tag;this.children=[];this.dataset={};this.attributes={};this._text="";this.className="";this.classes=new Set();
    this.classList={toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name),add:name=>this.classes.add(name)};
  }
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(node=>node.textContent).join("");}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;this._text="";}
  setAttribute(key,value){this.attributes[key]=value;}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const context={
  document:{createElement:tag=>new Element(tag)},$:get,ui:{},
  setText:(id,value)=>get(id).textContent=value,
  number:(x,fallback=0)=>Number.isFinite(Number(x))?Number(x):fallback,
  displayLabel:x=>({pear:"梨",green:"绿色方块",apple:"苹果",banana:"香蕉"})[x]||x,
};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf("function renderBatch("),source.indexOf("function renderGraspFeedback(")),context);
vm.runInContext(source.slice(source.indexOf("function selectedId("),source.indexOf("function renderState(")),context);
let checks=0;
const eq=(a,b)=>{assert.equal(a,b);checks++;};
const has=(a,b)=>{assert.match(a,b);checks++;};

context.renderBatch({batch:{status:"idle",total:0}});
eq(get("batch-panel").hidden,true);
const items=[
  {index:1,label:"apple",zone_name:"zone_red",status:"succeeded",object_id:"old-apple",message:"已落位"},
  {index:2,label:"banana",zone_name:"zone_red",status:"executing",object_id:"new-banana",message:"正在下降"},
  {index:3,label:"pear",zone_name:"zone_green",status:"pending",object_id:null,message:""},
];
const state={
  status:"executing",busy:true,batch:{status:"running",total:3,completed:1,current_index:2,items,message:""},
  selected_object_id:"old-apple",last_result:{object_id:"old-apple"},
  messages:[{role:"assistant",decision:{action:"pick_batch",items:[{object_id:"old-apple"}]}}],
  scene:{objects:[{object_id:"new-banana",label:"banana"},{object_id:"old-apple",label:"apple"}]},
  simulation:{objects:[{id:"truth-apple",label:"apple"},{id:"truth-banana",label:"banana"}]},
};
context.renderBatch(state);
eq(get("batch-panel").hidden,false);
eq(get("batch-panel").dataset.status,"running");
eq(get("batch-state").textContent,"执行中");
eq(get("batch-progress").textContent,"1 / 3 件完成");
has(get("batch-message").textContent,/第 2 \/ 3 件：香蕉 · 抓放中/);
const rows=get("batch-items").children;
eq(rows.length,3);
eq(rows.map(row=>row.children[0].textContent).join(","),"1,2,3");
eq(rows[0].children[1].children[0].textContent,"苹果 → 红色区域");
eq(rows[1].children[1].children[0].textContent,"香蕉 → 红色区域");
eq(rows[2].children[1].children[0].textContent,"梨 → 绿色区域");
eq(rows[1].attributes["aria-current"],"step");
eq(rows[0].attributes["aria-current"],undefined);
eq(rows[1].children[2].textContent,"抓放中");
eq(rows[1].children[1].children[1].textContent,"正在下降");
eq(context.selectedId(state),"new-banana");
eq(context.selectedTruthId(state),"truth-banana");

items[1].status="observing";items[1].object_id="old-banana";
context.renderBatch(state);
eq(context.selectedId(state),"");
eq(context.selectedTruthId(state),"");
eq(get("batch-items").children[1].children[2].textContent,"重新观测");
items[1].status="recovering";items[1].object_id="new-banana";
context.renderBatch(state);
eq(context.selectedId(state),"new-banana");
eq(get("batch-items").children[1].children[2].textContent,"恢复中");

state.batch.status="failed";items[1].status="failed";items[1].message="恢复后仍未成功";
items[2].status="cancelled";state.batch.message="香蕉未完成，已取消后续任务";
context.renderBatch(state);
eq(context.selectedId(state),"");
eq(get("batch-state").textContent,"已中止");
eq(get("batch-progress").textContent,"1 / 3 件完成");
eq(get("batch-items").children[1].dataset.status,"failed");
eq(get("batch-items").children[2].children[2].textContent,"已取消");
eq(get("batch-message").textContent,"香蕉未完成，已取消后续任务");
eq(get("batch-items").children[1].classes.has("current"),false);

state.batch.status="stopped";state.batch.current_index=null;state.batch.message="";
context.renderBatch(state);
eq(context.selectedId(state),"");
eq(get("batch-state").textContent,"已停止");
has(get("batch-message").textContent,/后续物品不会继续执行/);
state.batch.status="complete";state.batch.completed=3;
items[1].status="succeeded";items[2].status="skipped";
context.renderBatch(state);
eq(context.selectedId(state),"");
eq(get("batch-state").textContent,"全部完成");
eq(get("batch-progress").textContent,"3 / 3 件完成");
eq(get("batch-items").children[2].children[2].textContent,"已在目标区");
context.renderBatch({batch:{status:"idle",total:0}});
eq(get("batch-panel").hidden,true);
eq(context.selectedId({status:"executing",messages:[{role:"assistant",decision:{object_id:"single"}}]}),"single");
eq(context.selectedId({last_result:{object_id:"single-result"}}),"single-result");

const html=fs.readFileSync(path.join(staticRoot,"index.html"),"utf8");
has(html,/data-instruction="依次把苹果和香蕉放到红色区域"/);
has(html,/data-instruction="把所有物品放到各自默认区域"/);
has(html,/正常批量任务仅请求一次模型生成计划/);
has(html,/每件最多尝试 3 次，失败即停止后续物品/);
has(html,/暂不支持堆叠、单件反复搬运多次/);
eq(html.includes("暂不支持多目标"),false);
for(const id of ["batch-panel","batch-state","batch-progress","batch-message","batch-items"])
  eq((html.match(new RegExp('id="'+id+'"',"g"))||[]).length,1);
console.log("Batch UI mock DOM: "+checks+" assertions passed.");
