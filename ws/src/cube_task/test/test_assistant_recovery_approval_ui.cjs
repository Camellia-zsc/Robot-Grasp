/* Real approval listeners in a small DOM; no browser, service or model calls. */
"use strict";
const fs=require("fs"),path=require("path"),vm=require("vm"),assert=require("assert/strict");
const staticRoot=path.join(__dirname,"..","cube_task","assistant","static");
const source=fs.readFileSync(path.join(staticRoot,"app.js"),"utf8");
const html=fs.readFileSync(path.join(staticRoot,"index.html"),"utf8");
class Element{
  constructor(tag="div"){
    this.tagName=tag;this.children=[];this.dataset={};this.attributes={};this._text="";this.value="";this.listeners=new Map();this.classes=new Set();
    this.classList={toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name)};
  }
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>child.textContent).join("");}
  set innerHTML(_value){throw new Error("Approval content must use textContent");}
  replaceChildren(...children){this.children=children;this._text="";}
  addEventListener(type,callback){const list=this.listeners.get(type)||[];list.push(callback);this.listeners.set(type,list);}
  async dispatch(type,fields={}){
    const event={target:this,preventDefault(){},...fields};
    await Promise.all((this.listeners.get(type)||[]).map(callback=>callback(event)));
  }
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const copy=value=>JSON.parse(JSON.stringify(value));
const requests=[],samples=[new Element()],speeds=[new Element()];speeds[0].dataset.speed="1";
const ui={state:null,connected:true,pending:false};
let response=async()=>({ok:true});
const context={
  document:{createElement:tag=>new Element(tag),querySelectorAll:selector=>selector==="[data-speed]"?speeds:selector==="[data-instruction]"?samples:[]},
  window:{},ui,$:get,
  setText:(id,value)=>get(id).textContent=value,
  number:(value,fallback=0)=>Number.isFinite(Number(value))?Number(value):fallback,
  showError:message=>{get("inline-error").textContent=message||"";},
  renderState:state=>{ui.state=copy(state);context.updateControls();},
  api:async(url,payload)=>{requests.push({url,payload:payload===undefined?undefined:copy(payload)});return response(url,payload);},
};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf("function updateControls("),source.indexOf("// Decode only received characters")),context);
let checks=0;
const eq=(actual,expected)=>{assert.equal(actual,expected);checks++;};
const has=(actual,expected)=>{assert.match(actual,expected);checks++;};
const deep=(actual,expected)=>{assert.deepEqual(copy(actual),copy(expected));checks++;};
const update=state=>context.renderState(state);
const waiting=(id="proposal-1",version=8)=>({status:"awaiting_approval",busy:true,recovery_approval:{status:"pending",request_id:id,scene_version:version,question:"是否先暂放香蕉再挪开黄色方块？",steps:["将香蕉安全暂放","移开黄色方块，完成原任务后放回"],message:"这会临时移动黄色方块。拒绝将停止本次任务。"}});
const accept=get("recovery-approval-accept"),decline=get("recovery-approval-decline"),panel=get("recovery-approval");

(async()=>{
  update({status:"ready",busy:false});
  eq(panel.hidden,true);eq(get("instruction").disabled,false);
  const state=waiting();update(state);
  eq(panel.hidden,false);eq(accept.disabled,false);eq(decline.disabled,false);
  eq(get("recovery-approval-question").textContent,state.recovery_approval.question);
  eq(get("recovery-approval-message").textContent,state.recovery_approval.message);
  eq(get("recovery-approval-steps").children.length,2);
  eq(get("recovery-approval-steps").children[1].textContent,state.recovery_approval.steps[1]);
  has(get("recovery-approval-status").textContent,/等待你的明确选择/);
  eq(get("send-button").disabled,true);eq(get("instruction").disabled,true);eq(get("reset-button").disabled,true);
  eq(samples[0].disabled,true);eq(get("stop-button").disabled,false);
  has(get("command-hint").textContent,/操作已暂停/);

  // Even an inconsistent busy flag cannot turn chat text into approval or new work.
  update({...state,busy:false});
  eq(get("send-button").disabled,true);eq(get("stop-button").disabled,false);
  await context.submitInstruction("同意");await context.submitInstruction("把香蕉放到抓取区");
  eq(requests.length,0);

  let finish;
  response=()=>new Promise(resolve=>{finish=resolve;});
  const first=accept.dispatch("click");
  eq(requests.length,1);
  deep(requests[0],{url:"/api/recovery-approval",payload:{request_id:"proposal-1",scene_version:8,approved:true}});
  eq(accept.disabled,true);eq(decline.disabled,true);eq(get("stop-button").disabled,false);
  has(get("recovery-approval-status").textContent,/正在提交/);
  await accept.dispatch("click");await decline.dispatch("click");
  eq(requests.length,1);
  finish({ok:true});await first;
  has(get("recovery-approval-status").textContent,/已提交，等待处理/);
  eq(/已完成|恢复成功/.test(get("recovery-approval-status").textContent),false);
  update(state);await accept.dispatch("click");
  eq(requests.length,1);eq(accept.disabled,true);eq(decline.disabled,true);

  for(const status of ["approved","declined","stale","none"]){
    update({...state,status:"recovering",recovery_approval:{...state.recovery_approval,status}});
    eq(panel.hidden,true);
  }
  update({status:"ready",busy:false});eq(panel.hidden,true);

  // New proposals require their own explicit click, including a changed scene version.
  response=async()=>({ok:true});update(waiting("proposal-1",9));
  eq(accept.disabled,false);await decline.dispatch("click");
  deep(requests.at(-1),{url:"/api/recovery-approval",payload:{request_id:"proposal-1",scene_version:9,approved:false}});
  has(get("recovery-approval-status").textContent,/已提交，等待处理/);
  update(waiting("proposal-2",9));eq(accept.disabled,false);eq(decline.disabled,false);

  // Stale approval is rejected, then state is refreshed; the old action is never retried.
  const stale=waiting("proposal-2",9);stale.recovery_approval.status="stale";stale.status="recovering";
  response=async url=>{if(url==="/api/recovery-approval")throw Object.assign(new Error("场景已改变，原方案已失效"),{status:409});return stale;};
  const beforeStale=requests.length;await accept.dispatch("click");
  eq(requests.length,beforeStale+2);eq(requests.at(-1).url,"/api/state");
  eq(panel.hidden,true);has(get("inline-error").textContent,/方案已失效/);
  await accept.dispatch("click");eq(requests.length,beforeStale+2);

  // A refresh that still has the old pending snapshot must not re-enable a rejected proposal.
  const pending=waiting("proposal-3",10);update(pending);
  response=async url=>{if(url==="/api/recovery-approval")throw Object.assign(new Error("此方案不可再提交"),{status:400});return pending;};
  await decline.dispatch("click");
  eq(panel.hidden,false);eq(accept.disabled,true);eq(decline.disabled,true);
  has(get("recovery-approval-status").textContent,/不可再提交/);
  update(pending);const afterReject=requests.length;await decline.dispatch("click");
  eq(requests.length,afterReject);
  update(waiting("proposal-4",11));eq(accept.disabled,false);eq(decline.disabled,false);

  // Disconnection and state transitions cannot bypass the approval lock.
  ui.connected=false;context.updateControls();
  eq(accept.disabled,true);eq(decline.disabled,true);has(get("recovery-approval-status").textContent,/连接中断/);
  const locked=requests.length;await accept.dispatch("click");eq(requests.length,locked);
  ui.connected=true;context.updateControls();eq(accept.disabled,false);
  ui.pending=true;context.updateControls();eq(accept.disabled,true);
  await accept.dispatch("click");eq(requests.length,locked);ui.pending=false;
  const needsReset={...waiting("proposal-4",11),needs_reset:true};update(needsReset);
  eq(accept.disabled,true);await decline.dispatch("click");eq(requests.length,locked);
  update({...waiting("proposal-4",11),status:"stopping"});
  eq(accept.disabled,true);eq(get("stop-button").disabled,true);
  await accept.dispatch("click");eq(requests.length,locked);
  update({...waiting("proposal-4",11),status:"executing"});eq(accept.disabled,true);

  // Validate the request identity without parsing model text as markup or consent.
  const malformed=waiting("proposal-5",-1);update(malformed);
  eq(accept.disabled,true);await accept.dispatch("click");eq(requests.length,locked);
  update(waiting("",2));eq(accept.disabled,true);
  const escaped=waiting("proposal-6",0);
  escaped.recovery_approval.question="<img src=x onerror=approve()>";
  escaped.recovery_approval.steps=["<script>approve()</script>"];
  update(escaped);eq(accept.disabled,false);
  eq(get("recovery-approval-question").textContent,escaped.recovery_approval.question);
  eq(get("recovery-approval-steps").children[0].textContent,escaped.recovery_approval.steps[0]);
  eq(requests.length,locked);

  // Uncertain transport outcomes are refreshed and kept locked, never double submitted.
  response=async()=>{throw new Error("连接暂时不可用");};await accept.dispatch("click");
  eq(requests.at(-1).url,"/api/state");eq(accept.disabled,true);
  has(get("inline-error").textContent,/无法刷新状态/);
  const afterNetwork=requests.length;await accept.dispatch("click");eq(requests.length,afterNetwork);
  update(waiting("proposal-7",12));eq(accept.disabled,false);
  response=async()=>({ok:true});await get("stop-button").dispatch("click");
  eq(requests.at(-1).url,"/api/stop");

  has(html,/id="recovery-approval-accept"[^>]*>同意执行/);
  has(html,/id="recovery-approval-decline"[^>]*>拒绝并停止/);
  has(html,/id="recovery-approval-status"[^>]*aria-live="polite"/);

  const resume=get("resume-recovery-button"),resumePanel=get("resume-recovery");
  update({status:"error",busy:false,can_resume_recovery:false});eq(resumePanel.hidden,true);
  update({status:"recovering",busy:true,can_resume_recovery:true});eq(resumePanel.hidden,true);
  const held={status:"error",busy:false,can_resume_recovery:true,needs_reset:true,events:[{id:10,kind:"error"}]};
  update(held);eq(resumePanel.hidden,false);eq(resume.disabled,false);
  has(get("resume-recovery-status").textContent,/保留原目标，先分析恢复方案/);
  has(get("resume-recovery-status").textContent,/等待你点击“同意执行”/);
  has(get("command-hint").textContent,/继续处理当前持物/);
  const beforeResume=requests.length;
  ui.connected=false;context.updateControls();eq(resume.disabled,true);
  await resume.dispatch("click");eq(requests.length,beforeResume);
  ui.connected=true;ui.pending=true;context.updateControls();eq(resume.disabled,true);
  await resume.dispatch("click");eq(requests.length,beforeResume);
  ui.pending=false;context.updateControls();eq(resume.disabled,false);

  let resumeFinish;
  response=()=>new Promise(resolve=>{resumeFinish=resolve;});
  const resuming=resume.dispatch("click");
  eq(requests.length,beforeResume+1);
  deep(requests.at(-1),{url:"/api/resume-recovery",payload:{}});
  eq(resume.disabled,true);eq(get("stop-button").disabled,false);
  eq(get("send-button").disabled,true);eq(get("reset-button").disabled,true);
  has(get("resume-recovery-status").textContent,/正在提交/);
  await resume.dispatch("click");await context.submitInstruction("同意");
  eq(requests.length,beforeResume+1);
  resumeFinish({ok:true});await resuming;
  has(get("resume-recovery-status").textContent,/已提交，等待恢复分析/);
  eq(resume.disabled,true);update(held);await resume.dispatch("click");
  eq(requests.length,beforeResume+1);
  // Resume only asks for analysis. A later risky plan still needs its own approval click.
  update({status:"recovering",busy:true,can_resume_recovery:false});eq(resumePanel.hidden,true);
  update({...waiting("resumed-risk",13),can_resume_recovery:false});
  eq(panel.hidden,false);eq(accept.disabled,false);eq(resumePanel.hidden,true);
  eq(requests.length,beforeResume+1);

  const declined={...held,recovery_approval:{status:"declined",request_id:"resumed-risk"},events:[{id:11,kind:"result"}]};
  update(declined);eq(resumePanel.hidden,false);eq(resume.disabled,false);eq(panel.hidden,true);
  response=async()=>({ok:true});await resume.dispatch("click");
  deep(requests.at(-1),{url:"/api/resume-recovery",payload:{}});eq(resume.disabled,true);
  // Even if a short failed attempt fits between polls, a new terminal event enables an explicit retry.
  update({...declined,events:[{id:12,kind:"error"}]});eq(resume.disabled,false);
  const beforeFailure=requests.length;
  response=async url=>{if(url==="/api/resume-recovery")throw Object.assign(new Error("现场暂时无法继续处理"),{status:409});return {...declined,events:[{id:12,kind:"error"}]};};
  await resume.dispatch("click");
  eq(requests.length,beforeFailure+2);eq(requests.at(-1).url,"/api/state");
  eq(resume.disabled,false);has(get("resume-recovery-status").textContent,/现场暂时无法继续处理/);
  // Failed requests refresh, but never automatically retry or approve anything.
  eq(requests.slice(beforeFailure).some(request=>request.url==="/api/recovery-approval"),false);
  update({status:"stopped",busy:false,can_resume_recovery:false,needs_reset:true});
  eq(resumePanel.hidden,true);const stoppedRequests=requests.length;
  await resume.dispatch("click");eq(requests.length,stoppedRequests);
  update({status:"error",busy:false,can_resume_recovery:1});eq(resumePanel.hidden,true);
  update({...waiting("unrelated-pause",14),can_resume_recovery:true,busy:false});eq(resumePanel.hidden,true);
  await resume.dispatch("click");eq(requests.length,stoppedRequests);
  // A normal stop rearms only after the old worker exits; it then permits
  // editing/new commands and an explicit continuation without resetting.
  update({status:"stopped",busy:false,can_resume_recovery:true,needs_reset:false,
    simulation:{held_object:"pear"},events:[{id:30,kind:"stopped"}]});
  eq(resumePanel.hidden,false);eq(resume.disabled,false);
  eq(get("instruction").disabled,false);eq(get("send-button").disabled,false);
  response=async()=>({ok:true});
  await resume.dispatch("click");
  deep(requests.at(-1),{url:"/api/resume-recovery",payload:{}});
  has(html,/id="resume-recovery-button"[^>]*>继续处理当前持物/);
  has(html,/id="resume-recovery-status"[^>]*aria-live="polite"/);
  console.log(`Recovery approval UI: ${checks} assertions passed.`);
})().catch(error=>{console.error(error);process.exitCode=1;});
