/* Real skill-panel, pipeline, streaming and resume listeners; no service/model. */
"use strict";
const fs=require("fs"),path=require("path"),vm=require("vm"),assert=require("assert/strict");
const root=path.join(__dirname,"..","cube_task","assistant","static");
const source=fs.readFileSync(path.join(root,"app.js"),"utf8");
const html=fs.readFileSync(path.join(root,"index.html"),"utf8");
class Element{
  constructor(tag="div"){
    this.tagName=tag;this.children=[];this.dataset={};this.attributes={};this._text="";this.className="";this.classes=new Set();this.listeners=new Map();this.value="";
    this.scrollHeight=500;this.scrollTop=0;this.clientHeight=500;
    this.classList={toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name),add:name=>this.classes.add(name)};
  }
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>child.textContent).join("");}
  get childNodes(){return this.children;}
  set innerHTML(_){throw new Error("Model content must be rendered as text");}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;this._text="";}
  setAttribute(name,value){this.attributes[name]=value;}
  querySelector(){if(!this.number)this.number=new Element("span");return this.number;}
  addEventListener(type,callback){const list=this.listeners.get(type)||[];list.push(callback);this.listeners.set(type,list);}
  async dispatch(type,fields={}){await Promise.all((this.listeners.get(type)||[]).map(callback=>callback({target:this,preventDefault(){},...fields})));}
  remove(){}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const pipeline=Array.from({length:4},()=>new Element("li"));
const ui={connected:true,pending:false,messageNodes:new Map(),eventNodes:new Map()};
const requests=[],copy=value=>JSON.parse(JSON.stringify(value));
const context={document:{createElement:tag=>new Element(tag),querySelectorAll:selector=>selector==="#pipeline li"?pipeline:[]},window:{},ui,$:get,stateLabels:{},
  number:(value,fallback=0)=>Number.isFinite(Number(value))?Number(value):fallback,
  displayLabel:value=>({apple:"苹果",pear:"梨",banana:"香蕉"})[value]||String(value||"目标"),
  setText:(id,value)=>get(id).textContent=value,svgIcon:()=>new Element("svg"),
  showError:message=>get("inline-error").textContent=message||"",
  api:async(url,payload)=>{requests.push({url,payload:copy(payload)});return {ok:true};},
  renderState:state=>{ui.state=copy(state);context.updateControls();context.renderPipeline(state);context.renderMessages(state);},
};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf("function updateControls("),source.indexOf("// Decode only received characters")),context);
vm.runInContext(source.slice(source.indexOf("function readJsonString("),source.indexOf("function renderObservations(")),context);
vm.runInContext(source.slice(source.indexOf("function selectedId("),source.indexOf("function renderState(")),context);
let checks=0;
const eq=(actual,expected)=>{assert.equal(actual,expected);checks++;};
const has=(actual,expected)=>{assert.match(actual,expected);checks++;};
const update=state=>context.renderState(state);
const task={status:"executing",plan_number:2,max_plans:5,executed:3,max_executed:64,current_index:2,
  goals:[{label:"apple",zone_name:"zone_red"}],steps:[
    {id:"scan",skill:"observe",status:"succeeded"},
    {id:"hold",skill:"pick",label:"apple",status:"succeeded"},
    {id:"temp",skill:"place",label:"apple",destination:{type:"auto_free"},status:"executing",message:"正在寻找暂放空位"},
    {id:"clear",skill:"move",label:"pear",destination:{type:"zone",name:"zone_green"},status:"pending"},
    {id:"original",skill:"move",label:"pear",destination:{type:"original",label:"pear"},status:"pending"},
    {id:"goal",skill:"move",label:"apple",destination:{type:"goal",index:0},status:"pending"},
    {id:"region",skill:"place",label:"apple",destination:{type:"region",name:"pick_area"},status:"pending"},
    {id:"verify",skill:"verify",status:"pending"},
    {id:"ask",skill:"ask",question:"允许移动梨吗？",status:"pending"},
    {id:"finish",skill:"finish",status:"pending"}],history:[{skill:"pick",label:"apple",ok:true,message:"已抓取保持"}]};
const base={status:"executing",busy:true,skill_task:task,messages:[],events:[],
  batch:{status:"running",total:1,items:[]},recovery:{status:"executing",attempt:1,max_attempts:3},
  last_result:{ok:true,message:"暂放成功",truth:{success:true}},
  scene:{objects:[{label:"apple",object_id:"measured-apple",valid:true}]}};
(async()=>{
  update({status:"ready",busy:false,skill_task:{status:"idle"},messages:[]});
  eq(get("skill-panel").hidden,true);
  update(base);
  eq(get("skill-panel").hidden,false);eq(get("skill-panel").dataset.status,"executing");
  eq(get("batch-panel").hidden,true);eq(get("recovery-banner").hidden,true);
  eq(get("skill-state").textContent,"逐步执行中");has(get("skill-budget").textContent,/计划 2 \/ 5 · 已执行 3 \/ 64 步/);
  has(get("skill-goals").textContent,/最终目标：苹果 → 红色区域/);
  has(get("skill-message").textContent,/当前第 3 \/ 10 步：放置/);
  const rows=get("skill-steps").children;
  eq(rows.length,10);eq(rows[2].attributes["aria-current"],"step");eq(rows[2].children[2].textContent,"执行中");
  eq(rows[0].children[2].textContent,"步骤通过");has(rows[2].textContent,/实测空位（暂放）/);
  has(rows[3].textContent,/绿色区域/);has(rows[4].textContent,/梨原位置与朝向/);has(rows[5].textContent,/最终目标 1 · 红色区域/);
  has(rows[6].textContent,/抓取区/);has(rows[8].textContent,/询问人工.*允许移动梨吗/);has(rows[9].textContent,/最终核验/);
  eq(get("skill-feedback").children.length,1);has(get("skill-feedback").textContent,/抓取保持 · 苹果：已抓取保持/);
  eq(context.selectedId(base),"measured-apple");eq(get("task-result").hidden,true);

  // A successful temporary place/old last_result cannot complete original goals.
  update({...base,busy:false,status:"complete",skill_task:{...task,steps:task.steps.map(step=>({...step,status:"succeeded"}))}});
  eq(get("task-result").hidden,true);eq(pipeline.every(row=>row.classes.has("done")),false);
  const failed={...base,busy:false,status:"error",needs_reset:true,can_resume_recovery:true,
    skill_task:{...task,status:"failed",message:"原目标未完成：空位被占用",steps:task.steps.map((step,index)=>({...step,status:index===2?"failed":step.status})),
      history:[...task.history,{step:{skill:"place",label:"apple"},feedback:{ok:false,message:"空位被占用"}}]}};
  update(failed);eq(get("task-result").hidden,false);eq(get("task-result").classes.has("error"),true);
  has(get("task-result").textContent,/原目标未完成/);eq(get("skill-feedback").children.at(-1).dataset.status,"failed");
  has(get("skill-feedback").textContent,/放置 · 苹果：空位被占用/);eq(get("resume-recovery-button").textContent,"继续未完成任务");
  has(get("command-hint").textContent,/继续未完成任务/);has(get("resume-recovery-status").textContent,/保留未完成的原目标/);
  await get("resume-recovery-button").dispatch("click");eq(requests.length,1);eq(requests[0].url,"/api/resume-recovery");
  await get("resume-recovery-button").dispatch("click");eq(requests.length,1);

  update({...base,skill_task:{...task,status:"replanning",plan_number:3,current_index:null,message:"根据放置失败重新规划"}});
  eq(get("skill-state").textContent,"根据反馈重规划");eq(context.selectedId(ui.state),"");
  update({...base,busy:false,status:"complete",skill_task:{...task,status:"complete",current_index:null,message:"所有原目标已核验"}});
  eq(get("task-result").hidden,false);eq(get("task-result").classes.has("error"),false);
  has(get("task-result").textContent,/所有原目标已核验/);eq(pipeline.every(row=>row.classes.has("done")),true);

  // Skill event kinds remain visible beside real streaming text and render as text.
  const stream={id:"reply",role:"assistant",status:"streaming",content:'{"message":"先暂放苹果，再移开梨","steps":['};
  const events=["skill_plan","skill_step","skill_feedback","skill_replan"].map((kind,index)=>({id:index+1,kind,message:`真实反馈 ${index}`,ok:index===2?false:true}));
  update({...base,messages:[stream],events});
  const message=ui.messageNodes.get("reply");eq(message.body.textContent,"先暂放苹果，再移开梨");eq(message.stream.hidden,false);
  eq(ui.eventNodes.size,4);has(ui.eventNodes.get("3").className,/error/);
  update({...base,messages:[stream],events});eq(ui.eventNodes.size,4);
  update({...base,skill_task:{...task,message:"<img src=x onerror=run()>",history:[{skill:"place",ok:false,message:"<script>run()</script>"}]}});
  eq(get("skill-message").textContent,"<img src=x onerror=run()>");has(get("skill-feedback").textContent,/<script>run\(\)<\/script>/);
  for(const id of ["skill-panel","skill-state","skill-budget","skill-goals","skill-message","skill-steps","skill-feedback"])
    eq((html.match(new RegExp('id="'+id+'"',"g"))||[]).length,1);
  has(html,/暂放与单步通过不代表原任务完成/);has(html,/允许同一物品多次搬移/);
  console.log("Skill UI real rendering/listeners: "+checks+" assertions passed.");
})().catch(error=>{console.error(error);process.exitCode=1;});
