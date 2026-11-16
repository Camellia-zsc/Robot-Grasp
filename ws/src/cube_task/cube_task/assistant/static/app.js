/* All scene geometry and motion come from the server's simulation snapshots. */
"use strict";

const $ = (id) => document.getElementById(id);
const stateLabels = {preparing:"正在准备",editing:"正在更新摆场",ready:"场景就绪",observing:"相机观测中",thinking:"模型决策中",executing:"机械臂执行中",recovering:"自动恢复中",verifying:"结果验证中",complete:"任务已完成",stopping:"正在停止",stopped:"已停止",error:"执行异常"};
const labelNames = {red:"红色方块",red_cube:"红色方块",green:"绿色方块",green_cube:"绿色方块",yellow:"黄色方块",yellow_cube:"黄色方块",blue:"蓝色方块",blue_cube:"蓝色方块",apple:"苹果",banana:"香蕉",pear:"梨"};
const palette = {red:"#cc7771",green:"#7cab90",yellow:"#d1b76d",blue:"#789fc3",apple:"#cf766b",banana:"#d5b85f",pear:"#a9b76a"};
const ui = {state:null,connected:false,pending:false,messageNodes:new Map(),eventNodes:new Map(),trail:[],lastScene:null,lastCameraAt:0,lastCameraVersion:null,lastTime:null,cameraMode:"live",cameraViews:[],cameraKey:null,cameraLoading:false};
const canvas = $("scene-canvas");
const context = canvas.getContext("2d");
const view = {yaw:-1.08,pitch:0.68,zoom:1,width:800,height:397,dirty:true,drag:null,target:[0.30,0,0.14]};
let frameItems = [];
let right = [1,0,0], up = [0,0,1], facing = [0,-1,0], sceneScale = 400;

const number = (v,fallback=0) => Number.isFinite(Number(v)) ? Number(v) : fallback;
const vec = (v,fallback=[0,0,0]) => Array.isArray(v) && v.length>=3 && v.slice(0,3).every(n=>Number.isFinite(Number(n))) ? v.slice(0,3).map(Number) : fallback.slice();
const add = (a,b) => a.map((v,i)=>v+b[i]);
const sub = (a,b) => a.map((v,i)=>v-b[i]);
const mul = (a,n) => a.map(v=>v*n);
const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const length = a => Math.hypot(...a);
const normalize = a => mul(a,1/(length(a)||1));
function colorKey(label){const text=String(label||"").toLowerCase();return Object.keys(palette).find(k=>text.includes(k)) || (text.includes("红")?"red":text.includes("绿")?"green":text.includes("黄")?"yellow":text.includes("蓝")?"blue":null);}
function displayLabel(label){return labelNames[label] || labelNames[colorKey(label)] || String(label||"未分类目标");}
function setText(id,value){const text=String(value??"");if($(id).textContent!==text)$(id).textContent=text;}
function svgIcon(name){const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");const use=document.createElementNS(svg.namespaceURI,"use");use.setAttribute("href",`#i-${name}`);svg.append(use);return svg;}
function showError(message){$("inline-error").hidden=!message;setText("inline-error",message||"");}

async function api(path,body){
  const abort=new AbortController();const timer=setTimeout(()=>abort.abort(),12000);
  try{
    const response=await fetch(path,{method:body===undefined?"GET":"POST",headers:body===undefined?{}:{"Content-Type":"application/json"},body:body===undefined?undefined:JSON.stringify(body),cache:"no-store",signal:abort.signal});
    let result;try{result=await response.json();}catch{throw new Error(`服务返回了无法读取的响应 (${response.status})`);}
    if(!response.ok || result.ok===false && !result.simulation)throw new Error(result.error||result.message||`请求未成功 (${response.status})`);
    return result;
  }finally{clearTimeout(timer);}
}
function setConnection(connected){
  ui.connected=connected;$("connection").classList.toggle("offline",!connected);
  setText("connection-text",connected?"服务已连接":"连接中断");
}
function updateControls(){
  const state=ui.state||{};const busy=Boolean(state.busy||ui.pending);const ready=ui.connected&&state.status!=="preparing";
  $("send-button").disabled=busy||!ready||Boolean(state.needs_reset);$("instruction").disabled=busy||Boolean(state.needs_reset);
  $("reset-button").disabled=busy||!ready;$("stop-button").disabled=!state.busy||state.status==="stopping";
  document.querySelectorAll("[data-instruction]").forEach(button=>button.disabled=busy||!ready||Boolean(state.needs_reset));
  setText("command-hint",state.needs_reset?"任务已停止，请先重置场景":busy?"指令处理中，请观察执行流程":"Enter 发送 · Shift + Enter 换行");
  document.querySelectorAll("[data-speed]").forEach(button=>{button.classList.toggle("active",number(button.dataset.speed)===number(state.speed,1));button.disabled=!ready||ui.pending;});
}
async function submitInstruction(instruction){
  const text=String(instruction||"").trim();if(!text||ui.pending||ui.state?.busy||ui.state?.needs_reset)return;
  ui.pending=true;showError("");updateControls();
  try{await api("/api/command",{instruction:text});$("instruction").value="";}
  catch(error){showError(error.message);}
  finally{ui.pending=false;updateControls();}
}
$("command-form").addEventListener("submit",event=>{event.preventDefault();submitInstruction($("instruction").value);});
$("instruction").addEventListener("keydown",event=>{if(event.key==="Enter"&&!event.shiftKey&&!event.isComposing){event.preventDefault();submitInstruction(event.target.value);}});
document.querySelectorAll("[data-instruction]").forEach(button=>button.addEventListener("click",()=>submitInstruction(button.dataset.instruction)));
$("reset-button").addEventListener("click",async()=>{
  ui.pending=true;updateControls();showError("");
  try{await api("/api/reset",{});ui.trail=[];ui.lastTime=null;ui.lastCameraVersion=null;ui.cameraKey=null;}
  catch(error){showError(error.message);}finally{ui.pending=false;updateControls();}
});
$("stop-button").addEventListener("click",async()=>{try{await api("/api/stop",{});}catch(error){showError(error.message);}});
document.querySelectorAll("[data-speed]").forEach(button=>button.addEventListener("click",async()=>{try{await api("/api/speed",{speed:number(button.dataset.speed,1)});}catch(error){showError(error.message);}}));

// Decode only received characters. An unfinished escape stays invisible until
// its remaining characters arrive, so the displayed text never invents a token.
function readJsonString(text,start){
  let value="",index=start+1;
  const escapes={'"':'"',"\\":"\\","/":"/",b:"\b",f:"\f",n:"\n",r:"\r",t:"\t"};
  while(index<text.length){
    const char=text[index];
    if(char==='"')return {value,end:index+1,complete:true};
    if(char!=="\\"){
      const code=text.charCodeAt(index);
      if(code<32)return {value,end:index,complete:false,invalid:true};
      if(code>=0xd800&&code<=0xdbff){
        if(index+1>=text.length)return {value,end:index,complete:false};
        const low=text.charCodeAt(index+1);
        if(low>=0xdc00&&low<=0xdfff){value+=text.slice(index,index+2);index+=2;continue;}
      }
      value+=char;index++;continue;
    }
    if(index+1>=text.length)return {value,end:index,complete:false};
    const escape=text[index+1];
    if(escape!=="u"){
      if(!Object.hasOwn(escapes,escape))return {value,end:index,complete:false,invalid:true};
      value+=escapes[escape];index+=2;continue;
    }
    const hex=text.slice(index+2,index+6);
    if(hex.length<4)return {value,end:index,complete:false};
    if(!/^[0-9a-f]{4}$/i.test(hex))return {value,end:index,complete:false,invalid:true};
    const code=parseInt(hex,16);
    if(code>=0xd800&&code<=0xdbff){
      // Keep a split escaped Unicode pair together for display.
      const lowHex=text.slice(index+8,index+12);
      if(text.slice(index+6,index+8)==="\\u"&&/^[0-9a-f]{4}$/i.test(lowHex)){
        const low=parseInt(lowHex,16);
        if(low>=0xdc00&&low<=0xdfff){value+=String.fromCharCode(code,low);index+=12;continue;}
      }
      if(index+12>text.length)return {value,end:index,complete:false};
    }
    value+=String.fromCharCode(code);index+=6;
  }
  return {value,end:index,complete:false};
}
function extractStreamMessage(content){
  let depth=0,index=0;
  if(!/^\s*\{/.test(content))return "";
  while(index<content.length){
    const char=content[index];
    if(char==="{"||char==="["){depth++;index++;continue;}
    if(char==="}"||char==="]"){depth--;index++;continue;}
    if(char!=='"'){index++;continue;}
    const token=readJsonString(content,index);
    if(!token.complete)return "";
    let next=token.end;while(/\s/.test(content[next]||"")&&next<content.length)next++;
    if(depth===1&&token.value==="message"&&content[next]===":"){
      next++;while(/\s/.test(content[next]||"")&&next<content.length)next++;
      return content[next]==='"'?readJsonString(content,next).value:"";
    }
    index=token.end;
  }
  return "";
}
function assistantText(message,content,streaming){
  if(streaming)return extractStreamMessage(content);
  if(typeof message.decision?.message==="string")return message.decision.message;
  try{const decision=JSON.parse(content);return typeof decision?.message==="string"?decision.message:"";}
  catch{return "";}
}

function renderMessages(state){
  const conversation=$("conversation");const nearBottom=conversation.scrollHeight-conversation.scrollTop-conversation.clientHeight<75;
  const messages=Array.isArray(state.messages)?state.messages:[];
  $("welcome").hidden=messages.length>0;
  if(!messages.length && !$("welcome").childNodes.length){
    const icon=document.createElement("div");icon.className="welcome-icon";icon.append(svgIcon("cube"));
    const heading=document.createElement("h3");heading.textContent="从一句话开始";
    const description=document.createElement("p");description.textContent="告诉我你想抓取哪个物品，\n我会根据相机观测选择目标，并调用机械臂执行。";
    const note=document.createElement("div");note.className="welcome-note";note.textContent="观测场景 → 选择目标 → 执行与验证";
    $("welcome").append(icon,heading,description,note);
  }
  const activeKeys=new Set();
  messages.forEach((message,index)=>{
    const key=String(message.id??`${index}-${message.role}`);activeKeys.add(key);
    let node=ui.messageNodes.get(key);
    const role=message.role||"assistant";
    if(!node){
      node=document.createElement("article");node.className=`message ${role==="user"?"user":role==="system"?"system":"assistant"}`;
      const header=document.createElement("div");header.className="message-meta";
      const author=document.createElement("span");author.textContent=role==="user"?"你":role==="system"?"执行反馈":"抓取助手";
      const timestamp=document.createElement("span");timestamp.className="message-time";
      const time=message.time;if(typeof time==="string"){const date=new Date(time);if(!Number.isNaN(date.getTime()))timestamp.textContent=date.toLocaleTimeString("zh-CN",{hour:"2-digit",minute:"2-digit"});}
      header.append(author,timestamp);
      const body=document.createElement("div");body.className="message-body";
      const stream=document.createElement("div");stream.className="stream-indicator";stream.hidden=true;
      for(let i=0;i<3;i++)stream.append(document.createElement("i"));const streamText=document.createElement("span");streamText.textContent="接收模型回复";stream.append(streamText);
      const details=document.createElement("details");details.className="model-decision";details.hidden=true;
      const summary=document.createElement("summary");summary.textContent="查看模型决策";
      const raw=document.createElement("pre");raw.className="model-decision-raw";details.append(summary,raw);
      node.append(header,body,stream,details);node.author=author;node.summary=summary;node.body=body;node.stream=stream;node.details=details;node.raw=raw;ui.messageNodes.set(key,node);conversation.append(node);
    }
    const recoveryReply=role==="assistant"&&message.kind==="recovery";
    node.classList.toggle("recovery-reply",recoveryReply);
    node.author.textContent=recoveryReply?`恢复助手 · 第 ${Math.max(1,number(message.attempt,1))} 次尝试`:role==="user"?"你":role==="system"?"执行反馈":"抓取助手";
    node.summary.textContent=recoveryReply?"查看模型恢复决策":"查看模型决策";
    const content=String(message.content??message.text??"");
    const streaming=message.status==="streaming"||message.status==="pending"||message.status==="thinking";
    const primary=role==="assistant"?assistantText(message,content,streaming):content;
    node.body.hidden=role==="assistant"&&!primary;
    if(node.body.textContent!==primary)node.body.textContent=primary;
    node.stream.hidden=!(role==="assistant"&&streaming);
    node.details.hidden=role!=="assistant"||!content||(streaming&&!primary);
    if(role==="assistant"&&node.raw.textContent!==content)node.raw.textContent=content;
  });
  for(const [key,node] of ui.messageNodes)if(!activeKeys.has(key)){node.remove();ui.messageNodes.delete(key);}
  for(const event of Array.isArray(state.events)?state.events:[]){
    if(!["decision","result","error","stopped","recovery_start","recovery_result","recovery_exhausted"].includes(event.kind)||!event.message)continue;
    if(["decision","recovery_result"].includes(event.kind)&&messages.some(message=>message.role==="assistant"&&assistantText(message,String(message.content??message.text??""),false)===event.message))continue;
    const key=String(event.id??event.seq??`${event.time}-${event.message}`);
    if(ui.eventNodes.has(key))continue;
    const node=document.createElement("div");node.className=`event-bubble${["error","recovery_exhausted"].includes(event.kind)?" error":""}`;
    const text=document.createElement("span");text.textContent=event.message;
    node.append(svgIcon(event.kind==="decision"?"spark":event.kind==="result"?"check":event.kind.startsWith("recovery_")?"reset":"focus"),text);conversation.append(node);ui.eventNodes.set(key,node);
  }
  if(nearBottom)conversation.scrollTop=conversation.scrollHeight;
}

function stageIndex(state){
  if(!state.busy&&state.status!=="complete")return -1;
  const stage=String(state.stage||"").toLowerCase();
  if(/verif|result|valid|check|finish|complete|落位|验证/.test(stage))return 3;
  if(/扫描|重新观察|相机观测/.test(stage))return 0;
  if(/execut|grasp|pick|place|motion|move|执行|抓取/.test(stage))return 2;
  if(/think|decid|model|llm|plan|模型|决策/.test(stage))return 1;
  if(/observ|camera|percept|detect|观测/.test(stage))return 0;
  const recoveryStage={observing:0,thinking:1,executing:2,verifying:3}[state.recovery?.status];
  return {observing:0,thinking:1,executing:2,recovering:recoveryStage??1,verifying:3}[state.status]??-1;
}
function renderRecovery(state){
  const recovery=state.recovery||{};const status=String(recovery.status||"idle");
  const attempt=Math.max(0,number(recovery.attempt));const maximum=Math.max(attempt,number(recovery.max_attempts,3));
  const banner=$("recovery-banner");banner.hidden=status==="idle"||attempt===0;
  if(banner.hidden)return;
  const labels={observing:"重新检查",thinking:"模型分析",executing:"执行方案",verifying:"核验结果",succeeded:"恢复成功",exhausted:"恢复已结束",stopped:"已停止"};
  banner.dataset.status=status;banner.classList.toggle("error",status==="exhausted");banner.classList.toggle("success",status==="succeeded");
  setText("recovery-title",`自动恢复 · 第 ${attempt} / ${maximum} 次尝试`);
  const fallback={observing:"重新检查当前物体与放置区域。",thinking:"正在接收模型的恢复方案，回复会显示在右侧。",executing:"正在执行模型选择的恢复方案。",verifying:"正在检查目标是否落入指定区域。",succeeded:"目标已通过抓放和落位核验。",exhausted:"自动恢复已结束，请查看反馈后处理。",stopped:"恢复流程已停止，不会继续执行。"};
  setText("recovery-message",recovery.message||fallback[status]||"正在处理执行异常。");
  const reason=typeof recovery.reason==="string"?recovery.reason:"";
  $("recovery-reason").hidden=!reason;setText("recovery-reason",reason?`异常：${reason}`:"");
  setText("recovery-status",labels[status]||"处理中");
}
function renderPipeline(state){
  let index=stageIndex(state);const result=state.last_result||state.result;
  const truthSuccess=Boolean(result?.ok!==false&&result?.truth?.success!==false&&
    (result?.truth?.success===true||result?.success===true||result?.ok===true&&state.status==="complete"));
  const completed=!state.busy&&result&&truthSuccess;
  if(completed)index=4;
  document.querySelectorAll("#pipeline li").forEach((node,i)=>{
    node.classList.toggle("active",Boolean(state.busy)&&i===index);node.classList.toggle("done",index>i);
    node.querySelector(".step-number").textContent=index>i?"✓":String(i+1);
  });
  const events=Array.isArray(state.events)?state.events:[];const latest=events.at(-1);
  const text=state.message||state.stage||latest?.message||stateLabels[state.status]||"等待一条指令";
  setText("stage-message",text);$("stage-message").title=text;
  renderRecovery(state);
  renderGraspFeedback(state);
  const panel=$("task-result");panel.hidden=!result||Boolean(state.busy);
  if(result&&!state.busy){
    const failed=result.ok===false||result.truth?.success===false||state.status==="error";
    panel.classList.toggle("error",failed);panel.replaceChildren(svgIcon(failed?"focus":"check"));
    const description=document.createElement("span");
    description.textContent=result.message||result.summary||result.truth?.message||state.stage||(truthSuccess?"抓取与落位验证通过":failed?"本次任务未完成，请查看执行反馈":"模型回复已完成");
    panel.append(description);
  }
}
function renderGraspFeedback(state){
  const feedback=state.simulation?.feedback||{},plan=feedback.grasp_plan||{};
  const checks=Array.isArray(feedback.checks)?feedback.checks:[];
  const panel=$("grasp-feedback");if(!panel)return;
  panel.hidden=!Object.keys(plan).length&&!checks.length;
  const mm=value=>value!=null&&Number.isFinite(Number(value))?`${(Number(value)*1000).toFixed(1)} mm`:"—";
  const parts=[];
  if(plan.label)parts.push(displayLabel(plan.label));
  if([plan.length,plan.width,plan.height].some(v=>v!=null))parts.push(`测量长 / 宽 / 高 ${mm(plan.length)} / ${mm(plan.width)} / ${mm(plan.height)}`);
  if(plan.expected_width!=null)parts.push(`预计夹持宽度 ${mm(plan.expected_width)}`);
  if(plan.open_width!=null)parts.push(`张开 ${mm(plan.open_width)}`);
  if(plan.close_width!=null)parts.push(`闭合目标 ${mm(plan.close_width)}`);
  if(plan.closing_angle_deg!=null)parts.push(`夹持角 ${number(plan.closing_angle_deg).toFixed(1)}°`);
  setText("grasp-plan-feedback",parts.join(" · "));
  const nodes=checks.map(check=>{
    const li=document.createElement("li");li.className=check.ok===false?"failed":check.ok===true?"passed":"pending";
    const mark=document.createElement("span");mark.className="feedback-check-mark";mark.textContent=check.ok===false?"!":check.ok===true?"✓":"·";
    const text=document.createElement("span");text.textContent=check.message||({verify_close:"闭合后检查夹持",lift:"抬升后检查夹持",transit:"搬运后检查夹持",placement:"检查目标落位"})[check.stage]||check.stage||"等待检查";
    if(check.measured_width!=null){const detail=document.createElement("small");detail.textContent=`实测夹爪 ${mm(check.measured_width)}${check.expected_width!=null?` / 预计 ${mm(check.expected_width)}`:""}`;text.append(detail);}
    li.append(mark,text);return li;
  });
  $("grasp-checks").replaceChildren(...nodes);
}
function renderObservations(state){
  const scene=state.scene||{};const objects=Array.isArray(scene.objects)?scene.objects:[];
  const unconfirmed=objects.filter(object=>object.valid===false).length;
  setText("object-count",`${objects.length-unconfirmed} 个目标${unconfirmed?` · ${unconfirmed} 条待确认`:""}`);
  const diagnostics=Array.isArray(scene.observation_diagnostics)?scene.observation_diagnostics:[];
  setText("observation-note",`检测来自 RGB-D 扫描，包含工作区与放置区中的方块和水果。摆场后等待下条指令重新观测。${diagnostics.length?`已合并 ${diagnostics.length} 条重复或混合观测，可切换相机视角查看。`:""}`);
  const rejectionNames={touches_image_border:"画面截断",touches_roi_border:"范围边缘截断",touches_zone:"轮廓接触区域边界",occluded:"轮廓被遮挡",partial_object:"轮廓不完整",few_points:"深度点不足",too_small:"轮廓过小",too_large:"轮廓过大",too_tall:"高度异常",merged:"物体轮廓粘连",low_score:"识别把握不足"};
  const selected=selectedId(state);const rows=[];
  for(const object of objects){
    const uncertain=object.valid===false;
    const row=document.createElement("tr");
    const name=document.createElement("td");const wrapper=document.createElement("span");wrapper.className="object-name";
    const swatch=document.createElement("i");swatch.className=`swatch ${colorKey(object.label)||"blue"}`;
    const label=document.createElement("span");label.textContent=`${uncertain?"疑似":""}${displayLabel(object.label)}`;wrapper.append(swatch,label);name.append(wrapper);
    const confidence=document.createElement("td");confidence.textContent=object.score==null?"—":`${(number(object.score)*100).toFixed(1)}%`;
    const position=document.createElement("td");position.className="position";position.textContent=Array.isArray(object.center_base)?object.center_base.slice(0,3).map(n=>number(n).toFixed(3)).join(", "):"—";
    const status=document.createElement("td");const tag=document.createElement("span");tag.className="object-status";
    const isSelected=!uncertain&&String(object.object_id??object.id)===selected;
    const statusText=uncertain?`${rejectionNames[object.reject_reason]||"轮廓未确认"} · 待确认`:isSelected?"已选择":"可搬运";
    if(uncertain)tag.title="尚未确认这是独立物体，不参与抓取。可切换相机视角查看原始轮廓。";
    if(object.in_zone){
      const zone=object.zone_name||object.zone||object.placed_zone;
      const zoneName=typeof zone==="object"?zone?.label||zone?.name:zone;
      const area=({red:"红色区域",green:"绿色区域",yellow:"黄色区域",blue:"蓝色区域"})[colorKey(zoneName)]||zoneName||"已放置区域";
      tag.textContent=`${area} · ${statusText}`;tag.classList.add("in-zone");
    }else tag.textContent=statusText;
    tag.classList.toggle("selected",isSelected);
    status.append(tag);row.append(name,confidence,position,status);rows.push(row);
  }
  const signature=JSON.stringify([objects,selected]);
  if(ui.observationSignature!==signature){
    ui.observationSignature=signature;$("object-table-body").replaceChildren(...rows);
    if(!rows.length){const row=document.createElement("tr");const cell=document.createElement("td");cell.colSpan=4;cell.className="empty-table";cell.textContent="执行指令后显示相机检测结果";row.append(cell);$("object-table-body").append(row);}
  }
  renderCamera(state);
}
function frameStamp(stamp){
  if(stamp==null)return "等待相机帧";
  if(typeof stamp==="number"&&stamp<1e9)return `仿真时刻 ${stamp.toFixed(1)} s`;
  const date=new Date(typeof stamp==="number"&&stamp<1e12?stamp*1000:stamp);
  return Number.isNaN(date.getTime())?String(stamp):date.toLocaleTimeString("zh-CN",{hour12:false});
}
function renderCamera(state){
  const source=state.observation_views||state.scene?.observation_views||[];
  ui.cameraViews=(Array.isArray(source)?source:[]).map((view,index)=>({...view,id:String(view.id??view.view_id??view.view??index)}));
  const signature=JSON.stringify(ui.cameraViews.map(view=>[view.id,view.label]));
  if(signature!==ui.cameraOptionsSignature){
    ui.cameraOptionsSignature=signature;
    const live=document.createElement("option");live.value="live";live.textContent="实时腕部视角";
    const options=ui.cameraViews.map(view=>{const option=document.createElement("option");option.value=`observation:${view.id}`;option.textContent=view.label||`观测视角 ${view.id}`;return option;});
    $("camera-select").replaceChildren(live,...options);
    if(ui.cameraMode!=="live"&&!ui.cameraViews.some(view=>`observation:${view.id}`===ui.cameraMode)){ui.cameraMode="live";ui.cameraKey=null;}
    $("camera-select").value=ui.cameraMode;
  }
  const camera=state.camera||{};const observation=ui.cameraViews.find(view=>`observation:${view.id}`===ui.cameraMode);
  const isLive=!observation&&camera.mode==="live";
  setText("camera-caption",observation?`${observation.label||`观测视角 ${observation.id}`} · 冻结观测`:isLive?"实时腕部视角":"最近一次腕部观测");
  setText("camera-mode-badge",observation?"冻结观测":isLive?`实时 · ${number(camera.fps,3)}fps`:"观测快照");
  setText("camera-time",frameStamp(observation?(observation.sim_time??observation.stamp):isLive?camera.sim_time:(state.scene?.observed_at??state.scene?.timestamp)));
  const labels=observation?.detected_labels;
  const names=Array.isArray(labels)?[...new Set(labels.map(label=>displayLabel(typeof label==="object"?label.label:label)))]:null;
  setText("camera-targets",observation?(names?names.length?`该视角检测到：${names.join("、")}`:"该视角未检测到目标":"该视角未提供检测摘要"):"实时画面仅供观察，不会逐帧更新检测列表");
  setText("camera-note",observation?"冻结画面来自最近一次任务扫描，可切换视角查看已放置区域。":isLive?"画面跟随机械臂运动；目标列表在任务扫描完成后更新。":"实时预览尚未就绪；当前显示最近一次相机观测。");
  $("camera-image").alt=observation?`${observation.label||"腕部相机"}的冻结观测` : isLive?"实时腕部相机预览":"最近一次腕部相机观测";
  $("camera-status-dot").classList.toggle("live",isLive&&Boolean(camera.available));
  refreshCamera();
}
function refreshCamera(){
  const state=ui.state;if(!state||!ui.connected||ui.cameraLoading)return;
  const camera=state.camera||{},observation=ui.cameraViews.find(view=>`observation:${view.id}`===ui.cameraMode);
  const now=Date.now();let key,url;
  if(observation){
    key=JSON.stringify([observation.id,state.scene?.scene_version,observation.sim_time??observation.stamp]);
    if(key===ui.cameraKey||observation.available===false)return;
    url=`/api/camera.jpg?mode=observation&view=${encodeURIComponent(observation.id)}&v=${encodeURIComponent(state.scene?.scene_version??"")}`;
  }else if(camera.mode==="live"){
    if(!camera.available||now-ui.lastCameraAt<350)return;
    key="live";url=`/api/camera.jpg?mode=live&t=${now}`;
  }else{
    if(state.camera_available===false||!state.scene?.scene_version)return;
    key=`legacy:${state.scene.scene_version}`;if(key===ui.cameraKey)return;
    url=`/api/camera.jpg?mode=observation&v=${encodeURIComponent(state.scene.scene_version)}`;
  }
  ui.cameraKey=key;ui.lastCameraAt=now;ui.cameraLoading=true;$("camera-image").src=url;
}
$("camera-select").addEventListener("change",event=>{
  ui.cameraMode=event.target.value;ui.cameraKey=null;ui.cameraLoading=false;ui.lastCameraAt=0;
  $("camera-image").hidden=true;$("camera-placeholder").hidden=false;
  if(ui.state)renderCamera(ui.state);
});
$("camera-image").addEventListener("load",()=>{ui.cameraLoading=false;$("camera-image").hidden=false;$("camera-placeholder").hidden=true;});
$("camera-image").addEventListener("error",()=>{ui.cameraLoading=false;$("camera-image").hidden=true;$("camera-placeholder").hidden=false;setText("camera-placeholder-text","暂时无法读取相机画面");});
function cameraRefreshLoop(){refreshCamera();setTimeout(cameraRefreshLoop,350);}
setTimeout(cameraRefreshLoop,350);
function selectedId(state){
  const result=state.last_result||state.result||{};
  const reply=state.status==="executing"?(state.messages||[]).findLast(message=>message.role==="assistant"&&message.decision):null;
  return String(state.selected_object_id??result.selected_object_id??result.object_id??result.decision?.object_id??result.decision?.target_object_id??reply?.decision?.object_id??"");
}
function selectedTruthId(state){
  const id=selectedId(state);if(!id)return "";
  const truth=state.simulation?.objects||[];if(truth.some(object=>String(object.id??object.object_id)===id))return id;
  const observed=(state.scene?.objects||[]).find(object=>String(object.object_id??object.id)===id);
  if(!observed)return "";
  const matching=truth.filter(object=>object.label===observed.label);
  return matching.length===1?String(matching[0].id??matching[0].object_id):"";
}
function renderState(state){
  if(ui.networkError){showError("");ui.networkError=false;}
  if(ui.backendError&&!state.error){showError("");ui.backendError=false;}
  ui.state=state;setConnection(true);updateControls();
  const model=state.model||state.provider||{};
  const modelParts=[model.provider,model.model].filter(Boolean);
  setText("provider-info",modelParts.join(" / ")||"尚未配置在线模型");$("provider-info").title=modelParts.join(" / ");
  setText("provider-status",model.configured?"已配置":"待配置");setText("model-live","");
  const liveDot=document.createElement("i");liveDot.className=model.configured?"configured":"";const liveText=document.createElement("span");liveText.textContent=model.configured?(state.status==="thinking"||state.recovery?.status==="thinking"?"接收回复":"在线模型"):"待配置";$("model-live").replaceChildren(liveDot,liveText);
  const statusText=stateLabels[state.status]||state.status||"场景就绪";setText("scene-state",statusText);
  $("scene-state-dot").className=state.status==="error"?"error":state.simulation?"active":"";
  if(state.error){showError(typeof state.error==="string"?state.error:JSON.stringify(state.error));ui.backendError=true;}
  renderMessages(state);renderPipeline(state);renderObservations(state);
  window.SceneEditor?.update(state);
  const sim=state.simulation;
  if(sim&&Array.isArray(sim.tcp_xyz)){
    $("canvas-empty").hidden=true;setText("sim-time",number(sim.sim_time).toFixed(1));
    const time=number(sim.sim_time);if(ui.lastTime!=null&&time<ui.lastTime)ui.trail=[];ui.lastTime=time;
    if(Array.isArray(sim.tcp_xyz)){
      const point=vec(sim.tcp_xyz);if(!ui.trail.length||length(sub(point,ui.trail.at(-1)))>.002){ui.trail.push(point);if(ui.trail.length>180)ui.trail.shift();}
    }
    view.dirty=true;
  }
  const metrics=state.metrics||{};
  setText("request-metrics",number(metrics.requests)>0?`${number(metrics.requests)} 次指令 · ${number(metrics.successes)} 次成功` :"等待实验");
}
async function poll(){
  try{renderState(await api("/api/state"));}
  catch(error){setConnection(false);updateControls();setText("scene-state","服务连接中断");$("scene-state-dot").className="error";if(!ui.state){showError("暂时无法连接实验台服务，正在重试…");ui.networkError=true;}}
  finally{setTimeout(poll,200);}
}

// Dependency-free orthographic orbit renderer. World units are metres.
function project(point){const d=sub(point,view.target);return [view.width*.5+dot(d,right)*sceneScale,view.height*.54-dot(d,up)*sceneScale,dot(point,facing)];}
function shade(hex,amount){const value=parseInt(hex.slice(1),16);return `rgb(${[value>>16&255,value>>8&255,value&255].map(v=>Math.max(0,Math.min(255,Math.round(v*amount)))).join(",")})`;}
function polygon(points,fill,stroke=null,width=.6){const projected=points.map(project);frameItems.push({depth:projected.reduce((s,p)=>s+p[2],0)/projected.length,draw(){context.beginPath();projected.forEach((p,i)=>i?context.lineTo(p[0],p[1]):context.moveTo(p[0],p[1]));context.closePath();if(fill){context.fillStyle=fill;context.fill();}if(stroke){context.lineWidth=width;context.strokeStyle=stroke;context.stroke();}}});}
function line(points,color,width=1,dash=[]){const projected=points.map(project);frameItems.push({depth:projected.reduce((s,p)=>s+p[2],0)/projected.length+.0005,draw(){context.beginPath();projected.forEach((p,i)=>i?context.lineTo(p[0],p[1]):context.moveTo(p[0],p[1]));context.strokeStyle=color;context.lineWidth=width;context.lineCap="round";context.setLineDash(dash);context.stroke();context.setLineDash([]);}});}
function box(center,size,color,yaw=0,axes=null,stroke="rgba(50,70,76,.08)"){
  const basis=axes||[[Math.cos(yaw),Math.sin(yaw),0],[-Math.sin(yaw),Math.cos(yaw),0],[0,0,1]];
  const vertices=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]].map(sign=>add(center,add(add(mul(basis[0],sign[0]*size[0]/2),mul(basis[1],sign[1]*size[1]/2)),mul(basis[2],sign[2]*size[2]/2))));
  const faces=[[0,4,7,3],[1,2,6,5],[0,1,5,4],[3,7,6,2],[0,3,2,1],[4,5,6,7]];
  const light=normalize([-.6,-.8,1.8]);
  for(const ids of faces){const points=ids.map(i=>vertices[i]);const normal=normalize(cross(sub(points[1],points[0]),sub(points[2],points[0])));if(dot(normal,facing)>0)polygon(points,shade(color,.76+.25*Math.max(0,dot(normal,light))),stroke);}
}
function cylinder(start,end,radius,color){
  const delta=sub(end,start);if(length(delta)<.0005)return;
  const axis=normalize(delta);const tangent=normalize(cross(axis,Math.abs(axis[2])>.9?[1,0,0]:[0,0,1]));const binormal=normalize(cross(axis,tangent));
  const rings=[[],[]];const count=12;
  for(let i=0;i<count;i++){const angle=i/count*Math.PI*2;const offset=add(mul(tangent,Math.cos(angle)*radius),mul(binormal,Math.sin(angle)*radius));rings[0].push(add(start,offset));rings[1].push(add(end,offset));}
  const light=normalize([-.6,-.8,1.8]);
  for(let i=0;i<count;i++){const j=(i+1)%count;const normal=normalize(sub(add(rings[0][i],rings[0][j]),mul(start,2)));if(dot(normal,facing)>0)polygon([rings[0][i],rings[0][j],rings[1][j],rings[1][i]],shade(color,.68+.35*Math.max(0,dot(normal,light))),"rgba(57,76,84,.06)",.4);}
  if(dot(axis,facing)>0)polygon(rings[1],shade(color,1.02),"rgba(57,76,84,.1)");else polygon(rings[0],shade(color,.8),"rgba(57,76,84,.1)");
}
function sphere(point,radius,color){const p=project(point),r=radius*sceneScale;frameItems.push({depth:p[2]+radius*.4,draw(){const gradient=context.createRadialGradient(p[0]-r*.3,p[1]-r*.4,r*.1,p[0],p[1],r);gradient.addColorStop(0,shade(color,1.08));gradient.addColorStop(.6,color);gradient.addColorStop(1,shade(color,.73));context.beginPath();context.arc(p[0],p[1],r,0,Math.PI*2);context.fillStyle=gradient;context.fill();context.strokeStyle="rgba(48,72,81,.12)";context.lineWidth=.6;context.stroke();}});}
function floorEllipse(point,rx,ry,color){const points=[];for(let i=0;i<28;i++){const angle=i/28*Math.PI*2;points.push(add(point,[Math.cos(angle)*rx,Math.sin(angle)*ry,0]));}polygon(points,color);}
function label(point,text,color="#7d939c",background="#ffffffdd",selected=false){
  const p=project(point);frameItems.push({depth:100,draw(){context.font=`${selected?"600 ":""}10px "Segoe UI","Microsoft YaHei",sans-serif`;const width=context.measureText(text).width+17,height=23,x=p[0]-width/2,y=p[1]-height;
  context.beginPath();context.roundRect(x,y,width,height,5);context.fillStyle=background;context.fill();context.strokeStyle=selected?"#9ac3b8":"#dfe8eb";context.lineWidth=.7;context.stroke();context.fillStyle=color;context.textAlign="center";context.textBaseline="middle";context.fillText(text,p[0],y+height/2);}});
}
function normalizedZones(sim,state){
  const zones=sim.zones||state.scene?.zones||[];
  return Array.isArray(zones)?zones:Object.entries(zones).map(([name,z])=>({...z,name}));
}
function objectPosition(object){return vec(object.center||object.position||object.center_base);}
function objectSize(object){
  if(Array.isArray(object.dimensions))return vec(object.dimensions,[.03,.03,.03]);
  const size=object.size||[.03,.03,.03];
  if(object.shape==="sphere")return Array(3).fill(2*number(size[0],.015));
  if(object.shape==="ellipsoid")return vec(size,[.015,.015,.015]).map(v=>2*v);
  if(object.shape==="cylinder")return [2*number(size[0],.015),2*number(size[0],.015),number(size[1],.03)];
  return typeof size==="number"?[size,size,size]:vec(size,[.03,.03,.03]);
}
function rgbHex(rgb,fallback="#9daeb8"){
  return Array.isArray(rgb)&&rgb.length===3?"#"+rgb.map(v=>Math.max(0,Math.min(255,Math.round(number(v)))).toString(16).padStart(2,"0")).join(""):fallback;
}
// Expand the same bottom-origin primitives used by SimObject. Both views use
// these world-space parts, so an editor preview never substitutes a cube.
function objectPrimitives(object){
  const axes=matrixAxes(object.transform)||[[Math.cos(number(object.yaw)),Math.sin(number(object.yaw)),0],[-Math.sin(number(object.yaw)),Math.cos(number(object.yaw)),0],[0,0,1]];
  const transform=object.transform;
  const origin=Array.isArray(transform)&&Array.isArray(transform[0])?[number(transform[0][3]),number(transform[1][3]),number(transform[2][3])]:sub(objectPosition(object),[0,0,objectSize(object)[2]/2]);
  const point=local=>add(origin,axes.reduce((out,axis,i)=>add(out,mul(axis,number(local[i]))),[0,0,0]));
  const color=rgbHex(object.color_rgb,palette[colorKey(object.label)]||"#9daeb8");
  const parts=[];
  function primitive(kind,size,position=[0,0,0],yaw=0,partColor=color,centered=true){
    const c=Math.cos(yaw),s=Math.sin(yaw),basis=[add(mul(axes[0],c),mul(axes[1],s)),add(mul(axes[0],-s),mul(axes[1],c)),axes[2]];
    let center=point(position);
    if(!centered){const h=kind==="sphere"?size[0]:kind==="ellipsoid"?size[2]:kind==="box"?size[2]/2:kind==="cylinder"?size[1]/2:0;center=add(center,mul(basis[2],h));}
    if(kind==="capsule"){
      parts.push({kind:"capsule",start:add(center,mul(basis[0],-size[1])),end:add(center,mul(basis[0],size[1])),radius:size[0],color:partColor});
    }else if(kind==="cylinder"){
      parts.push({kind:"cylinder",start:add(center,mul(basis[2],-size[1]/2)),end:add(center,mul(basis[2],size[1]/2)),radius:size[0],color:partColor});
    }else parts.push({kind,center,size,axes:basis,color:partColor});
  }
  const size=Array.isArray(object.size)?object.size:objectSize(object);
  if(object.shape==="compound"){
    for(const part of size){if(!part||!Array.isArray(part.size))continue;primitive(part.shape,part.size,vec(part.pos),number(part.yaw),part.color_bgr?rgbHex([...part.color_bgr].reverse(),color):color);}
  }else if(object.shape==="capsule_chain"||object.shape==="sphere_cluster"){
    const radius=number(size[0],.01),points=Array.isArray(size[1])?size[1].map(p=>point(vec(p))):[];
    points.forEach(center=>parts.push({kind:"sphere",center,size:[radius],axes,color}));
    if(object.shape==="capsule_chain")for(let i=1;i<points.length;i++)parts.push({kind:"cylinder",start:points[i-1],end:points[i],radius,color});
  }else primitive(object.shape||"box",size,[0,0,0],0,color,false);
  return parts;
}
function ellipsoid(center,radii,axes,color){
  const rings=8,segments=14,light=normalize([-.6,-.8,1.8]);
  const surface=(latitude,longitude)=>add(center,axes.reduce((out,axis,i)=>add(out,mul(axis,[radii[0]*Math.cos(latitude)*Math.cos(longitude),radii[1]*Math.cos(latitude)*Math.sin(longitude),radii[2]*Math.sin(latitude)][i])),[0,0,0]));
  for(let ring=0;ring<rings;ring++)for(let segment=0;segment<segments;segment++){
    const a=-Math.PI/2+ring/rings*Math.PI,b=-Math.PI/2+(ring+1)/rings*Math.PI,c=segment/segments*Math.PI*2,d=(segment+1)/segments*Math.PI*2;
    const middle=[Math.cos((a+b)/2)*Math.cos((c+d)/2)/radii[0],Math.cos((a+b)/2)*Math.sin((c+d)/2)/radii[1],Math.sin((a+b)/2)/radii[2]];
    const normal=normalize(axes.reduce((out,axis,i)=>add(out,mul(axis,middle[i])),[0,0,0]));
    if(dot(normal,facing)>0)polygon([surface(a,c),surface(a,d),surface(b,d),surface(b,c)],shade(color,.72+.33*Math.max(0,dot(normal,light))));
  }
}
function drawObjectShape(object){
  for(const part of objectPrimitives(object)){
    if(part.kind==="box")box(part.center,part.size,part.color,0,part.axes);
    else if(part.kind==="sphere")sphere(part.center,part.size[0],part.color);
    else if(part.kind==="ellipsoid")ellipsoid(part.center,part.size,part.axes,part.color);
    else if(part.kind==="cylinder"||part.kind==="capsule"){
      cylinder(part.start,part.end,part.radius,part.color);
      if(part.kind==="capsule"){sphere(part.start,part.radius,part.color);sphere(part.end,part.radius,part.color);}
    }
  }
}
function objectGeometryBounds(object){
  const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
  for(const part of objectPrimitives(object))for(let axis=0;axis<3;axis++){
    let lo,hi;
    if(part.start){const direction=normalize(sub(part.end,part.start)),radius=part.radius*(part.kind==="capsule"?1:Math.sqrt(Math.max(0,1-direction[axis]**2)));lo=Math.min(part.start[axis],part.end[axis])-radius;hi=Math.max(part.start[axis],part.end[axis])+radius;}
    else{const ext=part.kind==="sphere"?part.size[0]:part.kind==="box"?part.axes.reduce((s,basis,i)=>s+Math.abs(basis[axis])*part.size[i]/2,0):Math.sqrt(part.axes.reduce((s,basis,i)=>s+(basis[axis]*part.size[i])**2,0));lo=part.center[axis]-ext;hi=part.center[axis]+ext;}
    low[axis]=Math.min(low[axis],lo);high[axis]=Math.max(high[axis],hi);
  }
  return {low,high};
}
window.RobotGeometry={primitives:objectPrimitives,dimensions:objectSize,bounds:objectGeometryBounds};
function drawTable(sim,state){
  const tableZ=number(sim.table_z);const bounds=sim.table_bounds||{};
  const minX=number(bounds.min_x,-.14),maxX=number(bounds.max_x,.75),minY=number(bounds.min_y,-.48),maxY=number(bounds.max_y,.48);
  box([(minX+maxX)/2,(minY+maxY)/2,tableZ-.018],[maxX-minX,maxY-minY,.036],"#e4eaec");
  flushGeometry();
  polygon([[minX,minY,tableZ+.0001],[maxX,minY,tableZ+.0001],[maxX,maxY,tableZ+.0001],[minX,maxY,tableZ+.0001]],"#edf2f3","#d8e2e5",1);
  flushGeometry();
  for(let x=Math.ceil(minX*20)/20;x<=maxX;x+=.05)line([[x,minY,tableZ+.0002],[x,maxY,tableZ+.0002]],"#dce6e9",.55);
  for(let y=Math.ceil(minY*20)/20;y<=maxY;y+=.05)line([[minX,y,tableZ+.0002],[maxX,y,tableZ+.0002]],"#dce6e9",.55);
  for(const zone of normalizedZones(sim,state)){
    const center=vec(zone.center||zone.position);const size=zone.size||[.12,.12,.005];const sx=number(size[0],.12),sy=number(size[1],.12),z=tableZ+.0008;
    const name=zone.label||zone.name||"放置区域";const key=colorKey(name)||colorKey(zone.name)||Object.entries(state.scene?.class_to_zone||{}).find(([,value])=>value===zone.name)?.[0];const color=palette[colorKey(key)]||"#9db4ba";
    const points=[[center[0]-sx/2,center[1]-sy/2,z],[center[0]+sx/2,center[1]-sy/2,z],[center[0]+sx/2,center[1]+sy/2,z],[center[0]-sx/2,center[1]+sy/2,z]];
    polygon(points,`${color}1a`,`${color}88`,1);line([...points,points[0]],`${color}8a`,1,[4,4]);
    const zoneName=({red:"红色放置区",green:"绿色放置区",yellow:"黄色放置区",blue:"蓝色放置区"})[colorKey(key)]||String(name);
    label([center[0],center[1]+sy*.5+.025,z],zoneName,color,"#ffffffcc");
  }
  floorEllipse([0,0,tableZ+.001],.075,.067,"#667f8d12");
  const objects=Array.isArray(sim.objects)?sim.objects:[];
  for(const object of objects){const p=objectPosition(object),s=objectSize(object);const height=Math.max(0,p[2]-tableZ);floorEllipse([p[0]+height*.08,p[1]+height*.06,tableZ+.0012],s[0]*.63+height*.06,s[1]*.6+height*.04,object.held?"#607b8d0b":"#607b8d1c");}
}
function drawObjects(sim,state){
  for(const object of Array.isArray(sim.objects)?sim.objects:[]){
    const p=objectPosition(object),s=objectSize(object);const color=rgbHex(object.color_rgb,palette[colorKey(object.label)]||"#9daeb8");const selected=String(object.id??object.object_id)===selectedTruthId(state);
    if(selected){const z=number(sim.table_z)+.002;floorEllipse([p[0],p[1],z],s[0]*1.02,s[1]*1.02,"#57a88b1c");const points=[];for(let i=0;i<=36;i++){const a=i/36*Math.PI*2;points.push([p[0]+Math.cos(a)*s[0],p[1]+Math.sin(a)*s[1],z]);}line(points,"#6caa9699",1.1,[3,3]);}
    drawObjectShape(object);
    const suffix=object.held?" · 已夹持":object.placed?" · 已放置":"";
    label(add(p,[0,0,s[2]*.5+.028]),displayLabel(object.label)+suffix,selected?"#368e7b":"#82959d",selected?"#f5fcf8ed":"#ffffffdd",selected);
  }
}
function matrixAxes(transform){if(Array.isArray(transform)&&transform.length>=3&&Array.isArray(transform[0]))return [0,1,2].map(i=>normalize([number(transform[0][i]),number(transform[1][i]),number(transform[2][i])]));return null;}
function drawArm(sim){
  let points=Array.isArray(sim.link_frames)?sim.link_frames.map(frame=>Array.isArray(frame)&&Array.isArray(frame[0])?[frame[0][3],frame[1][3],frame[2][3]]:vec(frame.position||frame.translation)):[];
  if(!points.length&&Array.isArray(sim.link_points)){
    points=sim.link_points.map(p=>vec(p));
    if(points.length>2&&Array.isArray(sim.tcp_xyz)&&length(sub(points.at(-1),sim.tcp_xyz))<.0001)points.pop();
  }
  if(!points.length)return;
  if(length(points[0])>.015)points.unshift([0,0,number(sim.table_z)]);
  const baseZ=number(sim.table_z);
  cylinder([0,0,baseZ+.003],[0,0,baseZ+.032],.057,"#657b87");
  cylinder([0,0,baseZ+.032],[0,0,baseZ+.061],.046,"#e7edef");
  const nonzero=[];
  for(let i=1;i<points.length;i++)if(length(sub(points[i],points[i-1]))>.004)nonzero.push([points[i-1],points[i],i]);
  for(const [start,end,index] of nonzero){
    // Endpoints are the actual kinematic link frames, never a procedural pose.
    const radius=index<=2?.025:index<=4?.020:.016;
    cylinder(start,end,radius,index<=3?"#d7e1e5":"#bdccd3");
    const direction=normalize(sub(end,start));const offset=mul(direction,Math.min(length(sub(end,start))*.08,.009));
    cylinder(add(start,offset),add(start,mul(offset,2)),radius*1.04,"#7c929d");
  }
  for(let i=1;i<points.length;i++)sphere(points[i],i<3?.030:i<5?.024:.019,i%2?"#c8d6dd":"#dfe7eb");
  const tcp=Array.isArray(sim.tcp_xyz)?vec(sim.tcp_xyz):points.at(-1);
  const transform=sim.tcp_transform||sim.tcp_pose;
  let axes=matrixAxes(transform)||matrixAxes(sim.tcp_rotation);
  if(!axes){const axis=normalize(sub(tcp,points.at(-2)||[0,0,0]));const x=normalize(cross(Math.abs(axis[2])>.9?[1,0,0]:[0,0,1],axis));axes=[x,normalize(cross(axis,x)),axis];}
  // Match sim.world.gripper_boxes: palm .045 × .09 × .09 m,
  // finger opening along tool Y and finger tips ending at the TCP plane.
  const palm=add(tcp,mul(axes[2],-.095));box(palm,[.045,.090,.090],"#63818d",0,axes);
  const width=Math.min(.12,Math.max(0,number(sim.gripper_width)));
  for(const side of [-1,1]){
    const finger=add(add(tcp,mul(axes[1],side*(width/2+.006))),mul(axes[2],-.025));
    box(finger,[.020,.012,.050],"#8ca5af",0,axes);box(add(finger,mul(axes[2],.018)),[.0205,.0125,.013],"#455e68",0,axes);
  }
  sphere(tcp,.0045,"#6cb7a4");
  if(ui.trail.length>1)line(ui.trail,"#73afa080",1.4,[3,5]);
  // A small marker names the true tool centre without competing with objects.
  const p=project(tcp);frameItems.push({depth:101,draw(){context.font='8px "Segoe UI",sans-serif';context.fillStyle="#80a99e";context.textAlign="left";context.fillText("TCP",p[0]+8,p[1]+2);}});
}
function drawAxisGizmo(){
  const origin=[view.width-42,view.height-48];const axes=[{p:[1,0,0],c:"#be8d85",t:"X"},{p:[0,1,0],c:"#85a695",t:"Y"},{p:[0,0,1],c:"#839fb9",t:"Z"}];
  for(const axis of axes){const x=origin[0]+dot(axis.p,right)*21,y=origin[1]-dot(axis.p,up)*21;context.beginPath();context.moveTo(...origin);context.lineTo(x,y);context.strokeStyle=axis.c;context.lineWidth=1.6;context.stroke();context.beginPath();context.arc(x,y,2.2,0,Math.PI*2);context.fillStyle=axis.c;context.fill();context.font='8px "Segoe UI",sans-serif';context.textAlign="center";context.fillText(axis.t,x+(x-origin[0])*.25,y+(y-origin[1])*.25+2);}
  context.beginPath();context.arc(...origin,2.4,0,Math.PI*2);context.fillStyle="#aebdc4";context.fill();
}
function flushGeometry(){frameItems.sort((a,b)=>a.depth-b.depth);for(const item of frameItems)item.draw();frameItems=[];}
function draw(){
  if(!view.dirty){requestAnimationFrame(draw);return;}view.dirty=false;
  const {width,height}=view;context.clearRect(0,0,width,height);
  const gradient=context.createLinearGradient(0,0,0,height);gradient.addColorStop(0,"#f6f8f9");gradient.addColorStop(1,"#edf2f4");context.fillStyle=gradient;context.fillRect(0,0,width,height);
  right=[-Math.sin(view.yaw),Math.cos(view.yaw),0];up=[-Math.cos(view.yaw)*Math.sin(view.pitch),-Math.sin(view.yaw)*Math.sin(view.pitch),Math.cos(view.pitch)];facing=[Math.cos(view.yaw)*Math.cos(view.pitch),Math.sin(view.yaw)*Math.cos(view.pitch),Math.sin(view.pitch)];
  sceneScale=Math.min(width/1.12,height/.77)*view.zoom;
  const state=ui.state||{};const sim=state.simulation;
  if(sim){frameItems=[];drawTable(sim,state);flushGeometry();drawArm(sim);drawObjects(sim,state);flushGeometry();drawAxisGizmo();
    const tcp=sim.tcp_xyz;if(Array.isArray(tcp)){context.font='9px "Consolas",monospace';context.fillStyle="#9aabb3";context.textAlign="left";context.fillText(`TCP  ${tcp.slice(0,3).map(n=>number(n).toFixed(3)).join("  ")} m`,20,height-47);}
  }
  requestAnimationFrame(draw);
}
function resize(){const rect=$("canvas-wrap").getBoundingClientRect();const dpr=Math.min(window.devicePixelRatio||1,2);view.width=rect.width;view.height=rect.height;canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);context.setTransform(dpr,0,0,dpr,0,0);view.dirty=true;}
new ResizeObserver(resize).observe($("canvas-wrap"));
canvas.addEventListener("pointerdown",event=>{view.drag={x:event.clientX,y:event.clientY,yaw:view.yaw,pitch:view.pitch};canvas.setPointerCapture(event.pointerId);canvas.classList.add("dragging");});
canvas.addEventListener("pointermove",event=>{if(!view.drag)return;view.yaw=view.drag.yaw-(event.clientX-view.drag.x)*.008;view.pitch=Math.max(.17,Math.min(1.35,view.drag.pitch+(event.clientY-view.drag.y)*.006));view.dirty=true;});
function endDrag(){view.drag=null;canvas.classList.remove("dragging");}
canvas.addEventListener("pointerup",endDrag);canvas.addEventListener("pointercancel",endDrag);
canvas.addEventListener("wheel",event=>{event.preventDefault();view.zoom=Math.max(.65,Math.min(2,view.zoom*Math.exp(-event.deltaY*.001)));view.dirty=true;},{passive:false});
$("view-reset").addEventListener("click",()=>{view.yaw=-1.08;view.pitch=.68;view.zoom=1;view.dirty=true;});
resize();requestAnimationFrame(draw);renderMessages({messages:[]});updateControls();poll();
