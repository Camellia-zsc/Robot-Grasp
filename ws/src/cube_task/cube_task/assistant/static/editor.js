/* Scene editing only. Robot targets continue to come from a later RGB-D scan. */
"use strict";

window.SceneEditor=(()=>{
  const byId=id=>document.getElementById(id);
  const map=byId("editor-canvas"),wrap=byId("editor-map-wrap"),ctx=map.getContext("2d");
  const editor={state:null,selected:null,chosen:null,drag:null,ghost:null,pending:false,pendingVersion:null,accepted:false,width:420,height:280,scale:350,center:[.3,0],catalogSignature:"",library:new Map(),yawEditing:false};
  const icons={red:"■",green:"■",yellow:"■",apple:"🍎",banana:"🍌",pear:"🍐"};
  const safeNumber=(v,fallback=0)=>Number.isFinite(Number(v))?Number(v):fallback;
  const objects=()=>Array.isArray(editor.state?.simulation?.objects)?editor.state.simulation.objects:[];
  const config=()=>editor.state?.simulation?.editor||{};
  const entries=()=>Array.isArray(config().catalog)?config().catalog:[];
  const selected=()=>objects().find(object=>String(object.id)===editor.selected);
  const position=object=>Array.isArray(object.center)?[safeNumber(object.center[0]),safeNumber(object.center[1])]:[safeNumber(object.transform?.[0]?.[3]),safeNumber(object.transform?.[1]?.[3])];
  const version=()=>editor.state?.simulation?.scene_version??config().scene_version;
  const canEdit=()=>Boolean(editor.state&&config().editable&&!editor.state.busy&&!editor.state.needs_reset&&!editor.pending&&!ui.pending&&ui.connected&&!editor.state.simulation?.held_object);
  const exists=entry=>objects().some(object=>object.label===entry.label);
  const project=point=>[editor.width/2+(point[0]-editor.center[0])*editor.scale,editor.height/2-(point[1]-editor.center[1])*editor.scale];
  const unproject=(x,y)=>[editor.center[0]+(x-editor.width/2)/editor.scale,editor.center[1]-(y-editor.height/2)/editor.scale];
  function pointer(event){const rect=map.getBoundingClientRect();return unproject((event.clientX-rect.left)*editor.width/rect.width,(event.clientY-rect.top)*editor.height/rect.height);}
  function setHint(text){byId("editor-hint").textContent=text;}
  function setError(text){byId("editor-error").hidden=!text;byId("editor-error").textContent=text||"";}
  function controls(){
    const available=canEdit(),object=selected();
    for(const entry of entries()){
      const button=editor.library.get(String(entry.id));if(!button)continue;
      const present=exists(entry),full=objects().length>=safeNumber(config().max_objects,6);
      button.disabled=!available||present||full;button.draggable=!button.disabled;
      button.classList.toggle("selected",editor.chosen===String(entry.id));
      button.classList.toggle("present",present);
      button.querySelector("small").textContent=present?"已在桌面":entry.description||"拖入抓取区";
    }
    byId("editor-delete").disabled=!available||!object;
    byId("editor-clear").disabled=!available||objects().length===0;
    byId("editor-yaw").disabled=!available||!object;
    byId("editor-selected-name").textContent=object?displayLabel(object.label):editor.chosen?entries().find(entry=>String(entry.id)===editor.chosen)?.name||"待放置物品":"未选中物品";
    byId("editor-selected-position").textContent=object?`X ${position(object)[0].toFixed(3)} · Y ${position(object)[1].toFixed(3)} m`:editor.chosen?"在虚线抓取区内点击放置":"点击或拖动物体";
    if(!editor.yawEditing){const degrees=object?Math.round(safeNumber(object.yaw)*180/Math.PI):0;byId("editor-yaw").value=Math.max(-180,Math.min(180,degrees));byId("editor-yaw-value").textContent=`${degrees}°`;}
    byId("editor-count").textContent=`${objects().length} / ${safeNumber(config().max_objects,6)} 件 · 每类最多一件`;
    const locked=!available;
    byId("editor-lock").hidden=!locked;
    byId("editor-lock").querySelector("span").textContent=editor.pending?"正在更新摆场…":editor.state?.busy?"任务运行中，摆场已锁定":editor.state?.simulation?.held_object?"夹持物体时不能编辑":editor.state?.needs_reset?"请先重置场景":!ui.connected?"等待服务连接":"正在准备可编辑场景";
    map.classList.toggle("editor-editable",available);map.setAttribute("aria-disabled",String(!available));
  }
  function buildLibrary(){
    const signature=JSON.stringify(entries());if(signature===editor.catalogSignature)return;
    editor.catalogSignature=signature;editor.library.clear();
    const nodes=entries().map(entry=>{
      const button=document.createElement("button");button.type="button";button.className="catalog-item";button.dataset.catalogId=String(entry.id);
      const icon=document.createElement("span");icon.className=`catalog-icon ${entry.label}`;icon.textContent=icons[entry.label]||"◇";
      const copy=document.createElement("span"),name=document.createElement("strong"),note=document.createElement("small");name.textContent=entry.name||displayLabel(entry.label);note.textContent="拖入抓取区";copy.append(name,note);button.append(icon,copy);
      button.title=`${entry.name||displayLabel(entry.label)}：拖到抓取区，或点击后在桌面放置`;
      button.addEventListener("click",()=>{if(!canEdit()||exists(entry))return;editor.selected=null;editor.chosen=String(entry.id);editor.ghost=null;setError("");setHint(`已选择${entry.name||displayLabel(entry.label)}，在虚线抓取区内点击放置。`);controls();draw();});
      button.addEventListener("dragstart",event=>{
        if(!canEdit()||exists(entry)){event.preventDefault();return;}
        editor.chosen=String(entry.id);editor.selected=null;event.dataTransfer.effectAllowed="copy";event.dataTransfer.setData("application/x-robot-catalog",String(entry.id));event.dataTransfer.setData("text/plain",String(entry.id));controls();
      });
      button.addEventListener("dragend",()=>{editor.ghost=null;draw();});
      editor.library.set(String(entry.id),button);return button;
    });
    byId("object-library").replaceChildren(...nodes);
  }
  async function edit(action,fields={}){
    if(!canEdit())return;
    editor.pending=true;editor.pendingVersion=version();editor.accepted=false;ui.pending=true;setError("");controls();updateControls();
    try{
      await api("/api/scene-edit",{scene_version:editor.pendingVersion,action,...fields});
      editor.accepted=true;editor.chosen=null;editor.ghost=null;editor.drag=null;
      setHint("正在更新布局；旧观测将失效，下一条抓取指令会重新扫描。");
      renderState(await api("/api/state"));
    }catch(error){editor.pending=false;editor.accepted=false;ui.pending=false;setError(error.message);setHint("修改未完成，请检查提示后重新摆放。");}
    finally{controls();updateControls();draw();}
  }
  function polygon(points,fill,stroke,width=1,dash=[]){
    if(!points.length)return;ctx.beginPath();points.map(project).forEach((point,index)=>index?ctx.lineTo(...point):ctx.moveTo(...point));ctx.closePath();
    if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
  }
  function drawPrimitive(part){
    ctx.fillStyle=part.color;ctx.strokeStyle="#506a6f33";ctx.lineWidth=.6;
    if(part.kind==="cylinder"||part.kind==="capsule"){
      const a=project(part.start),b=project(part.end),radius=Math.max(1,part.radius*editor.scale);
      ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle=part.color;ctx.lineWidth=radius*2;ctx.lineCap="round";ctx.stroke();ctx.lineCap="butt";
      // A vertical cylinder projects to a circle.
      if(Math.hypot(a[0]-b[0],a[1]-b[1])<.5){ctx.beginPath();ctx.arc(a[0],a[1],radius,0,Math.PI*2);ctx.fill();}
    }else if(part.kind==="box"){
      const points=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y])=>add(part.center,add(mul(part.axes[0],x*part.size[0]/2),mul(part.axes[1],y*part.size[1]/2))));polygon(points,part.color,"#506a6f33",.6);
    }else{
      const radii=part.kind==="sphere"?[part.size[0],part.size[0],part.size[0]]:part.size;
      const axes=part.axes||[[1,0,0],[0,1,0],[0,0,1]];
      // Projection of the ellipsoid covariance gives its exact XY ellipse.
      const a=axes.reduce((sum,axis,i)=>sum+(axis[0]*radii[i])**2,0),b=axes.reduce((sum,axis,i)=>sum+axis[0]*axis[1]*radii[i]**2,0),d=axes.reduce((sum,axis,i)=>sum+(axis[1]*radii[i])**2,0);
      const root=Math.sqrt((a-d)**2+4*b*b),major=Math.sqrt(Math.max(0,(a+d+root)/2)),minor=Math.sqrt(Math.max(0,(a+d-root)/2)),angle=-.5*Math.atan2(2*b,a-d),p=project(part.center);
      ctx.beginPath();ctx.ellipse(p[0],p[1],Math.max(.4,major*editor.scale),Math.max(.4,minor*editor.scale),angle,0,Math.PI*2);ctx.fill();ctx.stroke();
    }
  }
  function ghostObject(object,xy,yaw){
    const transform=[[Math.cos(yaw),-Math.sin(yaw),0,0],[Math.sin(yaw),Math.cos(yaw),0,0],[0,0,1,safeNumber(object.transform?.[2]?.[3])],[0,0,0,1]];
    const bounds=window.RobotGeometry.bounds({...object,transform,yaw});
    transform[0][3]=xy[0]-(bounds.low[0]+bounds.high[0])/2;transform[1][3]=xy[1]-(bounds.low[1]+bounds.high[1])/2;
    return {...object,transform,yaw};
  }
  function drawObject(object,alpha=1){ctx.save();ctx.globalAlpha=alpha;for(const part of window.RobotGeometry.primitives(object))drawPrimitive(part);ctx.restore();}
  function draw(){
    ctx.clearRect(0,0,editor.width,editor.height);ctx.fillStyle="#f4f8f7";ctx.fillRect(0,0,editor.width,editor.height);
    const sim=editor.state?.simulation||{},pick=config().pick_polygon||sim.pick_polygon||[];
    const zones=Array.isArray(sim.zones)?sim.zones:[];
    const points=pick.map(p=>[safeNumber(p[0]),safeNumber(p[1])]);
    for(const zone of zones){const center=zone.center||[0,0],size=zone.size||[.2,.2];points.push([center[0]-size[0]/2,center[1]-size[1]/2],[center[0]+size[0]/2,center[1]+size[1]/2]);}
    if(points.length){const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),minX=Math.min(...xs)-.045,maxX=Math.max(...xs)+.045,minY=Math.min(...ys)-.045,maxY=Math.max(...ys)+.045;editor.center=[(minX+maxX)/2,(minY+maxY)/2];editor.scale=Math.min((editor.width-22)/(maxX-minX),(editor.height-32)/(maxY-minY));}
    const lo=unproject(0,editor.height),hi=unproject(editor.width,0);
    ctx.strokeStyle="#dfe9e5";ctx.lineWidth=.5;
    for(let x=Math.ceil(lo[0]/.05)*.05;x<hi[0];x+=.05){const a=project([x,lo[1]]),b=project([x,hi[1]]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
    for(let y=Math.ceil(lo[1]/.05)*.05;y<hi[1];y+=.05){const a=project([lo[0],y]),b=project([hi[0],y]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
    for(const zone of zones){const [x,y]=zone.center||[0,0],[sx,sy]=zone.size||[.2,.2],yaw=safeNumber(zone.yaw),c=Math.cos(yaw),s=Math.sin(yaw),corners=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b])=>[x+a*sx/2*c-b*sy/2*s,y+a*sx/2*s+b*sy/2*c]);const color=palette[colorKey(zone.name)]||"#9cafad";polygon(corners,color+"12",color+"80",.8);const p=project([x,y]);ctx.fillStyle=color;ctx.font='9px "Microsoft YaHei",sans-serif';ctx.textAlign="center";ctx.fillText(({zone_red:"红区",zone_green:"绿区",zone_yellow:"黄区"})[zone.name]||zone.name,p[0],p[1]);}
    polygon(pick,"#60ab920a","#7dad9d",1,[5,4]);
    if(pick.length){const p=project([Math.min(...pick.map(p=>p[0]))+.01,Math.max(...pick.map(p=>p[1]))-.018]);ctx.font='9px "Microsoft YaHei",sans-serif';ctx.fillStyle="#7b9e90";ctx.textAlign="left";ctx.fillText("抓取区 · 拖入此处",p[0],p[1]);}
    for(const object of objects()){
      if(editor.drag?.objectId===String(object.id))continue;
      drawObject(object);
      if(String(object.id)===editor.selected){const p=project(object.center||[...position(object),0]),d=window.RobotGeometry.dimensions(object);ctx.beginPath();ctx.ellipse(p[0],p[1],Math.max(9,d[0]*editor.scale*.65),Math.max(9,d[1]*editor.scale*.65),0,0,Math.PI*2);ctx.strokeStyle="#428e79";ctx.lineWidth=1.5;ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);}
    }
    if(editor.ghost){
      if(editor.ghost.object){drawObject(ghostObject(editor.ghost.object,editor.ghost.xy,editor.ghost.yaw),.6);}
      else{const entry=entries().find(e=>String(e.id)===editor.chosen),p=project(editor.ghost.xy),dimensions=entry?.dimensions||[.06,.06,.06];ctx.fillStyle="#7fae9255";ctx.strokeStyle="#498e78";ctx.lineWidth=1;ctx.setLineDash([4,3]);ctx.beginPath();ctx.roundRect(p[0]-dimensions[0]*editor.scale/2,p[1]-dimensions[1]*editor.scale/2,dimensions[0]*editor.scale,dimensions[1]*editor.scale,4);ctx.fill();ctx.stroke();ctx.setLineDash([]);}
    }
    ctx.font='9px "Consolas",monospace';ctx.fillStyle="#8ca399";ctx.textAlign="left";ctx.fillText("X →   Y ↑  ·  网格 5 cm",12,editor.height-10);
  }
  function hit(xy){
    return [...objects()].reverse().find(object=>{const p=object.center||[...position(object),0],s=window.RobotGeometry.dimensions(object);return Math.abs(xy[0]-p[0])<=Math.max(s[0]/2,.014)&&Math.abs(xy[1]-p[1])<=Math.max(s[1]/2,.014);});
  }
  map.addEventListener("pointerdown",event=>{
    if(!canEdit()||event.button!==0)return;map.focus();const xy=pointer(event),object=hit(xy);setError("");
    if(object){editor.selected=String(object.id);editor.chosen=null;const p=position(object);editor.drag={objectId:String(object.id),object,start:xy,offset:[p[0]-xy[0],p[1]-xy[1]],xy:p,yaw:safeNumber(object.yaw),moved:false};map.setPointerCapture(event.pointerId);setHint(`已选中${displayLabel(object.label)}，拖动调整位置或使用右侧朝向。`);}
    else if(editor.chosen){const chosen=editor.chosen;edit("add",{catalog_id:chosen,xy,yaw:0});}
    else{editor.selected=null;setHint("拖动物品库卡片到抓取区，或点击已有物体移动。");}
    controls();draw();
  });
  map.addEventListener("pointermove",event=>{
    if(!canEdit())return;const xy=pointer(event);
    if(editor.drag){const drag=editor.drag;drag.xy=[xy[0]+drag.offset[0],xy[1]+drag.offset[1]];drag.moved=drag.moved||Math.hypot(xy[0]-drag.start[0],xy[1]-drag.start[1])>.002;editor.ghost={object:drag.object,xy:drag.xy,yaw:drag.yaw};draw();}
    else if(editor.chosen){editor.ghost={xy};draw();}
  });
  map.addEventListener("pointerup",event=>{
    if(!editor.drag)return;const drag=editor.drag;editor.drag=null;editor.ghost=null;if(map.hasPointerCapture(event.pointerId))map.releasePointerCapture(event.pointerId);
    if(drag.moved)edit("move",{object_id:drag.objectId,xy:drag.xy,yaw:drag.yaw});else draw();
  });
  map.addEventListener("pointercancel",()=>{editor.drag=null;editor.ghost=null;draw();});
  map.addEventListener("pointerleave",()=>{if(!editor.drag){editor.ghost=null;draw();}});
  map.addEventListener("dragover",event=>{if(!canEdit()||!editor.chosen)return;event.preventDefault();event.dataTransfer.dropEffect="copy";editor.ghost={xy:pointer(event)};draw();});
  map.addEventListener("drop",event=>{
    event.preventDefault();if(!canEdit())return;
    const catalogId=event.dataTransfer.getData("application/x-robot-catalog")||event.dataTransfer.getData("text/plain");
    if(!entries().some(entry=>String(entry.id)===catalogId)){setError("请选择物品库中的物品。");return;}
    edit("add",{catalog_id:catalogId,xy:pointer(event),yaw:0});
  });
  byId("editor-yaw").addEventListener("input",event=>{
    const object=selected();if(!canEdit()||!object)return;editor.yawEditing=true;const degrees=safeNumber(event.target.value);byId("editor-yaw-value").textContent=`${degrees}°`;editor.ghost={object,xy:position(object),yaw:degrees*Math.PI/180};draw();
  });
  byId("editor-yaw").addEventListener("change",event=>{const object=selected();editor.yawEditing=false;editor.ghost=null;if(object)edit("move",{object_id:String(object.id),xy:position(object),yaw:safeNumber(event.target.value)*Math.PI/180});});
  byId("editor-delete").addEventListener("click",()=>{const object=selected();if(object)edit("remove",{object_id:String(object.id)});});
  byId("editor-clear").addEventListener("click",()=>edit("clear"));
  map.addEventListener("keydown",event=>{
    if(event.key==="Escape"){editor.chosen=null;editor.selected=null;editor.ghost=null;editor.drag=null;controls();draw();return;}
    const object=selected();if(!canEdit()||!object)return;
    if(event.key==="Delete"||event.key==="Backspace"){event.preventDefault();edit("remove",{object_id:String(object.id)});return;}
    const deltas={ArrowLeft:[-.005,0],ArrowRight:[.005,0],ArrowUp:[0,.005],ArrowDown:[0,-.005]},delta=deltas[event.key];
    if(delta){event.preventDefault();const p=position(object);edit("move",{object_id:String(object.id),xy:[p[0]+delta[0],p[1]+delta[1]],yaw:safeNumber(object.yaw)});}
  });
  function resize(){const rect=wrap.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);editor.width=rect.width;editor.height=rect.height;map.width=Math.round(rect.width*dpr);map.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
  new ResizeObserver(resize).observe(wrap);
  function update(state){
    editor.state=state;
    if(editor.pending&&editor.accepted&&(version()!==editor.pendingVersion||!state.busy)){
      editor.pending=false;editor.accepted=false;ui.pending=false;
      if(state.error){setError(typeof state.error==="string"?state.error:"摆放失败，请调整位置后重试。");setHint("布局未改变，请查看错误提示。");}
      else setHint("布局已更新。发送抓取指令后，相机会重新扫描物品。");
      updateControls();
    }
    if(editor.selected&&!selected())editor.selected=null;
    if(editor.chosen){const entry=entries().find(item=>String(item.id)===editor.chosen);if(!entry||exists(entry)||objects().length>=safeNumber(config().max_objects,6)){editor.chosen=null;editor.ghost=null;}}
    if(state.busy&&!editor.pending){editor.drag=null;editor.ghost=null;editor.yawEditing=false;}
    buildLibrary();controls();draw();
  }
  resize();
  return {update};
})();
