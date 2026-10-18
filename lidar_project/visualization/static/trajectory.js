const $=id=>document.getElementById(id);
let index=0,playing=false,timer=null,pending=Promise.resolve();
const times=RUN.states.t, last=times.length-1, C=RUN.colors;
const alphaColor=(color,alpha)=>/^#[0-9a-f]{6}$/i.test(color)?`rgba(${parseInt(color.slice(1,3),16)},${parseInt(color.slice(3,5),16)},${parseInt(color.slice(5,7),16)},${alpha})`:color;
$('time').max=last;
$('run-info').textContent=`Run: ${RUN.name} | seed: ${RUN.seed} | final result: ${RUN.status} | ${last} control steps`;
const circle=(x,y,r)=>{const X=[],Y=[];for(let i=0;i<=40;i++){const a=2*Math.PI*i/40;X.push(x+r*Math.cos(a));Y.push(y+r*Math.sin(a));}return[X,Y];};
function pause(){playing=false;clearTimeout(timer);$('play').textContent='Play';}
function schedule(){if(!playing)return;if(index>=last){pause();return;}const delay=Math.max(16,1000*(times[index+1]-times[index])/(+$('speed').value));timer=setTimeout(()=>{index++;queueDraw().then(schedule);},delay);}
$('play').addEventListener('click',()=>{if(playing){pause();return;}if(index>=last)index=0;playing=true;$('play').textContent='Pause';queueDraw().then(schedule);});
$('previous').addEventListener('click',()=>{pause();index=Math.max(0,index-1);queueDraw();});
$('next').addEventListener('click',()=>{pause();index=Math.min(last,index+1);queueDraw();});
$('time').addEventListener('input',()=>{pause();index=+$('time').value;queueDraw();});
$('speed').addEventListener('change',()=>{if(playing){clearTimeout(timer);schedule();}});
['show-observations','show-scene','follow'].forEach(id=>$(id).addEventListener('change',()=>queueDraw()));
function queueDraw(){const k=index;pending=pending.catch(showError).then(()=>draw(k));return pending;}
function showError(error){$('status').textContent='The view could not be displayed: '+error.message;pause();}
async function draw(k){
 const xy=RUN.states.xy[k],theta=RUN.states.theta[k],t=times[k],hasScan=k<RUN.observations.length;
 $('time').value=k;$('time-label').textContent=`${t.toFixed(1)} s | ${k}/${last}`;
 const gap=RUN.states.surface_gap[k],v=hasScan?RUN.controls.v[k]:null,w=hasScan?RUN.controls.omega[k]:null;
 $('status').textContent=`True surface gap: ${gap==null?'none':gap.toFixed(2)+' m'} | `+(hasScan?
  `Speed command: ${v.toFixed(2)} m/s | Turn command: ${(w*180/Math.PI).toFixed(1)} deg/s | Returns: ${RUN.controls.n_returns[k]} | Obstacle points: ${RUN.controls.n_obstacle_points[k]}`:
  `Final state (${RUN.status}); no additional scan`);
 const traces=[];
 if($('show-scene').checked)for(const o of RUN.objects){
  traces.push({x:o.xy.map(p=>p[0]),y:o.xy.map(p=>p[1]),type:'scatter',mode:'lines',fill:'toself',
   line:{color:o.unknown?C.unknown:C.known_line,width:o.unknown?2:1},fillcolor:o.unknown?alphaColor(C.unknown,0.13):alphaColor(C.known,0.5),
   name:o.unknown?'Unmapped obstacle (truth)':'Prior-map obstacle (truth)',text:o.label,
   showlegend:false,hovertemplate:'%{text}<extra>ground-truth scene</extra>'});
 }
 traces.push({x:RUN.reference.map(p=>p[0]),y:RUN.reference.map(p=>p[1]),type:'scatter',mode:'lines',line:{color:C.reference,width:2,dash:'dash'},name:'Known-map reference',hoverinfo:'name'});
 traces.push({x:RUN.states.xy.slice(0,k+1).map(p=>p[0]),y:RUN.states.xy.slice(0,k+1).map(p=>p[1]),type:'scatter',mode:'lines',line:{color:C.actual,width:2.4},name:'Executed path',hovertemplate:'x=%{x:.2f} m<br>y=%{y:.2f} m<extra>executed path</extra>'});
 const start=RUN.reference[0],goal=RUN.reference[RUN.reference.length-1];
 traces.push({x:[start[0],goal[0]],y:[start[1],goal[1]],type:'scatter',mode:'markers',marker:{color:C.robot,size:[8,13],symbol:['circle','star']},text:['Start','Goal'],hoverinfo:'text',showlegend:false});
 if(hasScan&&$('show-observations').checked){const p=RUN.observations[k];traces.push({x:p.map(a=>a[0]),y:p.map(a=>a[1]),type:'scatter',mode:'markers',marker:{color:C.observations,size:3,opacity:0.8},name:'LiDAR obstacle returns',hovertemplate:'x=%{x:.2f} m<br>y=%{y:.2f} m<extra>noisy measurement</extra>'});}
 const [bx,by]=circle(xy[0],xy[1],RUN.radius);
 traces.push({x:bx,y:by,type:'scatter',mode:'lines',line:{color:C.robot,width:2},fill:'toself',fillcolor:alphaColor(C.robot,0.07),name:'Robot footprint',showlegend:false,hoverinfo:'name'});
 traces.push({x:[xy[0],xy[0]+Math.cos(theta)],y:[xy[1],xy[1]+Math.sin(theta)],type:'scatter',mode:'lines',line:{color:C.robot,width:3},showlegend:false,hoverinfo:'skip'});
 if(hasScan){const p=RUN.controls.target_xy[k];traces.push({x:[p[0]],y:[p[1]],type:'scatter',mode:'markers',marker:{color:C.target,size:8,symbol:'x'},name:'Lookahead target',showlegend:false,hoverinfo:'name'});}
 let xr=[RUN.extent[0],RUN.extent[1]],yr=[RUN.extent[2],RUN.extent[3]],revision='map';
 if($('follow').checked){xr=[xy[0]-5,xy[0]+5];yr=[xy[1]-3,xy[1]+3];revision='follow'+k;}
 const style={font:{family:RUN.font,color:C.actual,size:12},paper_bgcolor:C.paper,plot_bgcolor:C.plot};
 const mapLayout={...style,margin:{l:55,r:15,t:15,b:62},xaxis:{title:{text:'World x (m)'},range:xr,gridcolor:C.grid,zeroline:false},yaxis:{title:{text:'World y (m)'},range:yr,scaleanchor:'x',scaleratio:1,gridcolor:C.grid,zeroline:false},legend:{orientation:'h',x:0,y:-0.18,font:{size:10}},uirevision:revision};
 const commands=[
  {x:RUN.controls.t,y:RUN.controls.v,type:'scatter',mode:'lines',line:{color:C.speed,width:1.6},name:'Speed command',xaxis:'x',yaxis:'y',showlegend:false,hovertemplate:'t=%{x:.1f} s<br>v=%{y:.2f} m/s<extra></extra>'},
  {x:RUN.controls.t,y:RUN.controls.omega.map(w=>w*180/Math.PI),type:'scatter',mode:'lines',line:{color:C.turn,width:1.6},name:'Turn command',xaxis:'x2',yaxis:'y2',showlegend:false,hovertemplate:'t=%{x:.1f} s<br>turn=%{y:.1f} deg/s<extra></extra>'}];
 const commandsLayout={...style,margin:{l:62,r:16,t:15,b:50},xaxis:{domain:[0,1],anchor:'y',showticklabels:false,range:[0,times[last]],gridcolor:C.grid},yaxis:{domain:[0.58,1],title:{text:'Speed (m/s)'},gridcolor:C.grid},xaxis2:{domain:[0,1],anchor:'y2',title:{text:'Time (s)'},range:[0,times[last]],gridcolor:C.grid},yaxis2:{domain:[0,0.40],title:{text:'Turn (deg/s)'},gridcolor:C.grid},shapes:[{type:'line',xref:'x',yref:'paper',x0:t,x1:t,y0:0,y1:1,line:{color:C.actual,width:1,dash:'dot'}}],uirevision:'commands'};
 const options={responsive:true,displaylogo:false,scrollZoom:true};
 await Promise.all([Plotly.react('map',traces,mapLayout,options),Plotly.react('commands',commands,commandsLayout,options)]);
 document.documentElement.dataset.viewerReady='true';
}
window.addEventListener('resize',()=>queueDraw());
queueDraw();
