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
  displayLabel:value=>({apple:"苹果",pear:"梨",banana:"香蕉",red:"红色方块"})[value]||String(value||"目标"),
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

  // A support object is distinct from a colored destination zone. Physics
  // feedback explains settling and real drops instead of reporting completion.
  const stackDestination={type:"on_object",label:"red"};
  const stackTask={...task,current_index:0,goals:[{label:"apple",zone_name:null,destination:stackDestination}],
    steps:[{id:"stack",skill:"move",label:"apple",destination:stackDestination,status:"executing"}]};
  const physical={objects:[{id:"red-base",label:"red"},{id:"apple-top",label:"apple"}],
    physics:{backend:"mujoco",gravity:[0,0,-9.81],settling:true,support_relations:[],last_event:{}}};
  const stacking={...base,skill_task:stackTask,simulation:physical};
  update(stacking);
  has(get("skill-goals").textContent,/苹果 → 红色方块顶面/);
  has(get("skill-steps").textContent,/红色方块顶面/);
  eq(get("simulation-engine").textContent,"MuJoCo · 重力与接触");
  has(get("simulation-description").textContent,/真实重力接触/);
  eq(get("feedback-engine").textContent,"MuJoCo · 重力与接触");
  has(get("physics-feedback").textContent,/等待物品落稳.*尚不能判定叠放成功/);
  eq(get("physics-feedback").dataset.status,"settling");eq(get("task-result").hidden,true);
  update({...stacking,simulation:{...physical,physics:{...physical.physics,settling:false,
    support_relations:[{object:"apple-top",support:"red-base"}],last_event:{}}}});
  has(get("physics-feedback").textContent,/苹果由红色方块支撑/);
  const dropped={id:"real-drop",kind:"object_dropped",message:"苹果从红色方块滑落，正在重新观察"};
  update({...stacking,events:[dropped],simulation:{...physical,physics:{...physical.physics,settling:false,
    last_event:{kind:"object_dropped",message:dropped.message}}}});
  has(get("physics-feedback").textContent,/滑落/);eq(get("physics-feedback").dataset.status,"failed");
  has(ui.eventNodes.get("real-drop").className,/error/);eq(get("task-result").hidden,true);
  update(base);eq(get("feedback-engine").textContent,"运动学模拟检查");eq(get("physics-feedback").hidden,true);
  ui.eventNodes.get("real-drop").remove();ui.eventNodes.delete("real-drop");

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
  update({...failed,skill_task:{...failed.skill_task,
    goal_worklist:[{goal_index:0,label:"apple",status:"deferred",reason:"target_partial"}],
    steps:[{id:"deferred",skill:"move",label:"apple",destination:{type:"goal",index:0},status:"deferred"}],
    history:[{skill:"move",label:"apple",ok:false,deferred:true,message:"目标待确认，先处理其他物品"}]}});
  has(get("skill-goals").textContent,/暂缓确认：苹果（仍保留为最终目标）/);
  eq(get("skill-steps").children[0].children[2].textContent,"暂缓确认");
  eq(get("skill-feedback").children[0].dataset.status,"deferred");
  eq(get("skill-state").textContent,"任务尚未完成");
  update(failed);
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

  // Replay the banana hold: completed camera/placement checks are progress,
  // while the live held_object remains the source for current gripper state.
  const bananaTask={...task,plan_number:1,executed:0,current_index:0,message:"",history:[],
    goals:[{label:"banana",destination:{type:"region",name:"pick_area"}}],steps:[
      {id:"move",skill:"move",label:"banana",destination:{type:"goal",index:0},status:"executing",message:"正在执行"},
      {id:"finish",skill:"finish",status:"pending"}]};
  const held={held_object:"truth-banana",gripper_width:.040994,
    objects:[{id:"truth-banana",label:"banana",held:true}],
    feedback:{grasp_plan:{label:"banana",expected_width:.0413},checks:[
      {stage:"close",ok:true,message:"闭合后检测到夹持",measured_width:.041},
      {stage:"dropped",ok:true,message:"抬升后夹持检查",measured_width:.040994},
      {stage:"dropped_in_transit",ok:true,message:"搬运后夹持检查",measured_width:.040994}]}};
  const inspection={id:101,kind:"inspect_place",message:"已到目标区域上方，正在局部确认空位"};
  let placement={...base,skill_task:bananaTask,simulation:held,stage:inspection.message,events:[inspection]};
  update(placement);
  eq(ui.eventNodes.has("101"),true);has(get("skill-message").textContent,/局部确认空位/);
  has(get("grasp-plan-feedback").textContent,/仍夹持香蕉.*41\.0 mm.*尚未释放/);
  eq(get("task-result").hidden,true);
  const observed={id:102,kind:"recovery_observation",message:"已保持夹持并完成静止相机观测"};
  placement={...placement,stage:observed.message,events:[inspection,observed]};update(placement);
  eq(ui.eventNodes.has("102"),true);has(get("stage-message").textContent,/观测已完成.*正在检查安全放置/);
  has(get("skill-message").textContent,/正在检查安全放置/);
  const planning={id:103,kind:"placement_planning",message:"正在检查候选落点的支撑、碰撞和可达性"};
  placement={...placement,stage:planning.message,events:[...placement.events,planning]};update(placement);
  eq(ui.eventNodes.has("103"),true);has(get("stage-message").textContent,/检查候选落点/);
  const progress={id:104,kind:"placement_progress",message:"正在检查下一组安全放置候选"};
  placement={...placement,stage:progress.message,events:[...placement.events,progress]};update(placement);
  eq(ui.eventNodes.has("104"),true);has(get("skill-message").textContent,/下一组安全放置候选/);
  const refined={id:105,kind:"placement_refined",message:"已根据局部观测确认放置位置"};
  placement={...placement,stage:refined.message,events:[...placement.events,refined]};update(placement);
  eq(ui.eventNodes.has("105"),true);has(get("stage-message").textContent,/尚待释放与落位核验/);
  eq(get("task-result").hidden,true);eq(pipeline.every(row=>row.classes.has("done")),false);
  const changed={id:106,kind:"placement_changed",message:"当前测量下无法确认安全放置方案，将保持夹持"};
  placement={...placement,stage:changed.message,events:[...placement.events,changed]};update(placement);
  eq(ui.eventNodes.has("106"),true);has(get("skill-message").textContent,/无法确认安全放置/);
  const activeView={id:108,kind:"active_observation",message:"保持夹持，正在选择能看清目标区域的安全视角"};
  update({...placement,events:[...placement.events,activeView],stage:activeView.message});
  eq(ui.eventNodes.has("108"),true);has(get("stage-message").textContent,/安全视角.*仍夹持香蕉/);
  has(get("skill-message").textContent,/安全视角/);eq(get("task-result").hidden,true);

  // A status-only stop must invalidate the task's UI cache immediately.
  const stopping={...placement,status:"stopping",needs_reset:true,stage:"正在停止当前任务"};
  update(stopping);
  eq(get("skill-state").textContent,"正在停止");eq(get("skill-panel").dataset.status,"stopping");
  has(get("stage-message").textContent,/已请求停止.*等待当前计算/);
  has(get("skill-message").textContent,/仍夹持香蕉.*尚未释放/);
  eq(get("skill-steps").children[0].children[2].textContent,"停止中");
  eq(get("skill-steps").children[0].classes.has("current"),false);
  eq(pipeline.some(row=>row.classes.has("active")||row.classes.has("done")),false);
  has(get("command-hint").textContent,/等待当前计算/);
  eq(get("reset-button").disabled,true);eq(get("stop-button").disabled,true);
  eq(get("send-button").disabled,true);eq(get("resume-recovery").hidden,true);
  eq(get("task-result").hidden,true);
  update({...stopping,busy:false});eq(get("reset-button").disabled,true);eq(get("send-button").disabled,true);
  update(stopping);

  // Late success/placement events cannot override the controller's stop latch.
  const late={...stopping,stage:"已确认放置位置",message:"迟到的执行反馈",
    skill_task:{...bananaTask,status:"complete",message:"迟到的步骤通过"},
    events:[...stopping.events,{...refined,id:107}]};
  update(late);
  eq(get("skill-state").textContent,"正在停止");has(get("stage-message").textContent,/已请求停止/);
  eq(get("task-result").hidden,true);eq(pipeline.some(row=>row.classes.has("active")||row.classes.has("done")),false);
  const stopped={...late,status:"stopped",busy:false,stage:"迟到的完成反馈",message:"",can_resume_recovery:true};
  update(stopped);
  eq(get("skill-state").textContent,"已停止");eq(get("task-result").classes.has("error"),true);
  has(get("task-result").textContent,/任务已停止/);has(get("task-result").textContent,/仍夹持香蕉/);
  eq(pipeline.some(row=>row.classes.has("done")),false);eq(get("reset-button").disabled,false);
  eq(get("resume-recovery").hidden,true);
  has(get("command-hint").textContent,/请先重置场景/);
  const countBeforeStopResume=requests.length;await get("resume-recovery-button").dispatch("click");eq(requests.length,countBeforeStopResume);
  // A held-only change also refreshes the task cache, without claiming a place succeeded.
  update({...stopped,simulation:{...held,held_object:null}});
  eq(/仍夹持香蕉/.test(get("skill-message").textContent),false);
  eq(/仍夹持香蕉/.test(get("grasp-plan-feedback").textContent),false);
  eq(get("task-result").classes.has("error"),true);
  update({...stopped,stage:"任务已停止；仍有承诺的原位复位未完成",skill_task:{status:"idle"}});
  eq(get("skill-panel").hidden,true);has(get("stage-message").textContent,/原位复位未完成/);
  update({...placement,stage:refined.message,events:[refined],simulation:{...held,held_object:null}});
  eq(/尚待释放|仍夹持/.test(get("stage-message").textContent),false);
  has(get("stage-message").textContent,/选点确认不代表落位核验通过/);
  for(const id of ["skill-panel","skill-state","skill-budget","skill-goals","skill-message","skill-steps","skill-feedback"])
    eq((html.match(new RegExp('id="'+id+'"',"g"))||[]).length,1);
  has(html,/暂放与单步通过不代表原任务完成/);has(html,/允许同一物品多次搬移/);
  for(const id of ["simulation-engine","simulation-description","feedback-engine","physics-feedback"])
    eq((html.match(new RegExp('id="'+id+'"',"g"))||[]).length,1);
  has(html,/最近四轮对话/);has(html,/把苹果放到红色方块上面/);
  console.log("Skill UI real rendering/listeners: "+checks+" assertions passed.");
})().catch(error=>{console.error(error);process.exitCode=1;});
