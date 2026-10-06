"use strict";
const record = JSON.parse(document.getElementById("records").textContent);
const overlays = JSON.parse(document.getElementById("overlays").textContent);
const $ = id => document.getElementById(id);
const ns = "http://www.w3.org/2000/svg";
const axis = record.protocol.query_axis;
const models = [["kumo", "Kumo Tabular Small"], ["knn", "k-NN (3 neighbors)"], ["lr", "Logistic regression"]];
const charts = {};
let flip = 0;
function color(p) { const a = [12,25,42], b = [56,214,198]; return `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*p)).join(",")})`; }
function element(tag, attrs, parent, text) { const el = document.createElement(tag); Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v)); if(text !== undefined) el.textContent=text; parent.append(el); return el; }
function svg(tag, attrs, parent) { const el = document.createElementNS(ns,tag); Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v)); parent.append(el); return el; }
function xy(x,y) { return [120+x*80,120-y*80]; }
for(let i=0;i<10;i++) { const node=element("i",{},$("scale")); node.style.background=color(i/9); }
for(let y=3;y>=0;y--) for(let x=0;x<4;x++) {
  const i=y*4+x, q=record.context[i];
  const b=element("button",{"data-flip":i,"aria-pressed":"false","aria-label":`Flip context point ${i+1}, X ${q.x}, Y ${q.y}, original label ${q.inside}`},$("points"));
  element("span",{},b,`${q.x.toFixed(1)}, ${q.y.toFixed(1)}`); element("span",{},b,`${q.inside} → ${1-q.inside}`);
  b.addEventListener("click",()=>{flip=i;renderFlip();});
}
models.forEach(([key,name])=>{
  const article=element("article",{"class":"model","data-model":key},$("model-grid"));
  element("h3",{},article,name);
  const maps=element("div",{"class":"maps"},article);
  charts[key]=[];
  ["Before","After"].forEach((label,index)=>{
    const figure=element("figure",{"class":"map"},maps); element("figcaption",{},figure,label);
    const plot=svg("svg",{"class":"plot",viewBox:"0 0 240 240",role:"img","aria-labelledby":`${key}-${index}-title ${key}-${index}-desc`},figure);
    const title=svg("title",{id:`${key}-${index}-title`},plot); title.textContent=`${name}, ${label.toLowerCase()}, saved P(inside)`;
    svg("desc",{id:`${key}-${index}-desc`},plot).textContent="169 query cells, identical color scale and coordinates. Solid class boundaries, outlined and hatched changed cells, and a dashed farthest-reach ring in After maps. Context dots and a gold flipped-row circle. Reach rings can extend past the map edge. Use the query sliders for exact readouts.";
    const cells=record.queries.map((q,i)=>{
      const [cx,cy]=xy(q.x,q.y), cell=svg("rect",{x:cx-7.8,y:cy-7.8,width:15.6,height:15.6,"data-query":i},plot);
      cell.addEventListener("click",()=>{$("x").value=i%13;$("y").value=Math.floor(i/13);renderQuery();});return cell;
    });
    svg("circle",{cx:120,cy:120,r:48,fill:"none",stroke:"#eff6ff","stroke-dasharray":"1 4","stroke-opacity":.5},plot);
    const boundary=svg("g",{"class":"class-boundary"},plot);
    const boundaryDark=svg("path",{"class":"overlay-dark"},boundary);
    const boundaryLight=svg("path",{"class":"overlay-light"},boundary);
    const changes=svg("g",{"class":"changed-cells"},plot);
    const ring=svg("g",{"class":"reach-ring",display:"none"},plot);
    const ringDark=svg("circle",{"class":"overlay-dark"},ring);
    const ringLight=svg("circle",{"class":"overlay-light"},ring);
    const dots=record.context.map(q=>{const [cx,cy]=xy(q.x,q.y);return svg("circle",{cx,cy,r:3,fill:"none",stroke:"#eff6ff","stroke-width":1.5},plot);});
    const marker=svg("rect",{"class":"focus-marker",width:15.6,height:15.6},plot);
    charts[key].push({cells,dots,marker,title,boundary,boundaryDark,boundaryLight,changes,ring,ringDark,ringLight});
  });
  const dl=element("dl",{"class":"reach","data-reach":key},article);
  for(const [metric,label] of [["changed_predictions","Changed classes"],["max_probability_change","Largest probability shift"],["farthest_changed_distance","Farthest class change"]]) {
    element("dt",{},dl,label);element("dd",{id:`${key}-${metric}`},dl);
  }
  element("p",{"class":"notes"},article,"Reach for the selected flip, across all 169 queries.");
  const line=element("p",{},$("all-reaches"));
  line.textContent=`${name}: ${record.models[key].conditions.slice(1).map(c=>c.reach.changed_predictions).join(", ")}.`;
});
function entry(parent,key,value) { element("dt",{},parent,key);element("dd",{},parent,value); }
entry($("provenance"),"Kumo settings",`CPU float32 · 1 estimator · upstream default recipe · seed ${record.protocol.seed}`);
entry($("provenance"),"Comparators",`scikit-learn ${record.environment.sklearn} · k-NN(3), remaining defaults · LogisticRegression(), defaults · identical float32 features`);
entry($("provenance"),"Recorded execution",`${record.environment.cpu_model} · Python ${record.environment.python} · PyTorch ${record.environment.torch} · 2 CPU threads`);
entry($("provenance"),"Time / memory",`${record.total_record_seconds.toFixed(2)} s after imports, excluding setup; maximum RSS ${(record.maximum_rss_kib/1024).toFixed(0)} MiB. Container: 2 CPUs / 2 GiB / 180 s; no GPU, no inference network.`);
entry($("provenance"),"Baseline reuse",`Kumo before-map retained from v1. No v1 inference rerun. V1 fixed-repeat difference was 0; this is not independent reproduction.`);
entry($("provenance"),"Recorded UTC",`${record.started_at} → ${record.finished_at}`);
for(const [k,v] of Object.entries({"Upstream source revision":record.protocol.source_revision,"Model revision":record.protocol.model_revision,"Checkpoint SHA-256":record.weight_sha256,"Freeze SHA-256":record.freeze_sha256,"Protocol v2 SHA-256":record.input_hashes["protocol-v2.json"],"Recorder v2 SHA-256":record.input_hashes["record_v2.py"],"V2 dependency lock SHA-256":record.input_hashes["environment-v2/uv.lock"]})) entry($("hashes"),k,v);
models.forEach(([key,name])=>element("p",{"class":"query-line",id:`${key}-query`},$("query-values")));
function renderQuery() {
  const i=Number($("y").value)*13+Number($("x").value), q=record.queries[i];
  $("coordinate").textContent=`X ${q.x.toFixed(1)} · Y ${q.y.toFixed(1)} · query ${i+1} / 169 · synthetic label ${q.inside}`;
  for(const coordinate of ["x","y"]) { $(coordinate+"-value").textContent=q[coordinate].toFixed(1);$(coordinate).setAttribute("aria-valuetext",`${q[coordinate].toFixed(1)} synthetic units`); }
  models.forEach(([key,name])=>{
    const a=record.models[key].conditions[0].probabilities[i],b=record.models[key].conditions[flip+1].probabilities[i];
    const prediction=pair=>pair[1]>pair[0]?"inside":"outside";
    $(key+"-query").textContent=`${name}: P(inside) ${a[1].toFixed(3)} → ${b[1].toFixed(3)}; Δ ${((b[1]-a[1])*100).toFixed(1)} pp; ${prediction(a)} → ${prediction(b)}.`;
    const [cx,cy]=xy(q.x,q.y); charts[key].forEach(c=>{c.marker.setAttribute("x",cx-7.8);c.marker.setAttribute("y",cy-7.8);});
  });
}
function renderFlip() {
  const q=record.context[flip];
  document.querySelectorAll("[data-flip]").forEach(b=>b.setAttribute("aria-pressed",String(Number(b.dataset.flip)===flip)));
  $("selected-point").textContent=`Point ${flip+1} / 16: (${q.x.toFixed(1)}, ${q.y.toFixed(1)}), ${q.inside} → ${1-q.inside}.`;
  models.forEach(([key,name])=>{
    const cs=record.models[key].conditions;
    charts[key].forEach((chart,index)=>{
      const condition=cs[index===0?0:flip+1];
      const overlay=overlays[key][index===0?0:flip+1];
      chart.cells.forEach((cell,i)=>{cell.setAttribute("fill",color(condition.probabilities[i][1]));cell.setAttribute("data-class",overlay.classes[i]);});
      const path=overlay.boundary_edges.map(e=>`M${e[0]} ${e[1]}L${e[2]} ${e[3]}`).join(" ");
      chart.boundaryDark.setAttribute("d",path);chart.boundaryLight.setAttribute("d",path);
      chart.changes.replaceChildren();
      overlay.changed_cells.forEach(i=>{
        const q=record.queries[i], [cx,cy]=xy(q.x,q.y);
        svg("path",{"class":"changed-cell","data-changed-query":i,stroke:overlay.mark_inks[i],d:`M${cx-6} ${cy-6}h12v12h-12Z M${cx-5} ${cy+5}l10 -10`},chart.changes);
      });
      chart.ring.setAttribute("display",overlay.ring?"inline":"none");
      chart.ring.removeAttribute("data-radius-units");
      if(overlay.ring){
        chart.ring.setAttribute("data-radius-units",overlay.ring.radius_units);
        [chart.ringDark,chart.ringLight].forEach(circle=>{circle.setAttribute("cx",overlay.ring.cx);circle.setAttribute("cy",overlay.ring.cy);circle.setAttribute("r",overlay.ring.radius_px);});
      }
      chart.dots.forEach((dot,i)=>{dot.setAttribute("r",i===flip?6:3);dot.setAttribute("stroke",i===flip?"#ffd166":"#eff6ff");});
      chart.title.textContent=`${name}, ${index===0?"before":"after flip of point "+(flip+1)}, saved P(inside)`;
    });
    const r=cs[flip+1].reach;
    $(key+"-changed_predictions").textContent=`${r.changed_predictions} / 169`;
    $(key+"-max_probability_change").textContent=`${(r.max_probability_change*100).toFixed(1)} pp`;
    $(key+"-farthest_changed_distance").textContent=r.farthest_changed_distance===null?"none":`${r.farthest_changed_distance.toFixed(2)} units`;
  });renderQuery();
}
$("reset").addEventListener("click",()=>{flip=0;$("x").value=6;$("y").value=6;renderFlip();});
$("x").addEventListener("input",renderQuery);$("y").addEventListener("input",renderQuery);
renderFlip();
