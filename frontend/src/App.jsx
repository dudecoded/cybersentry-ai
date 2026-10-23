import { useEffect, useMemo, useState } from "react";
import {
  Activity, ArrowUpRight, CheckCircle2, ChevronDown,
  CircleHelp, Clock3, Cpu, Fingerprint, Gauge, Globe2, LockKeyhole,
  Menu, Radar, RefreshCw, ScanLine, Shield, ShieldAlert, ShieldCheck, Siren,
  Sparkles, Wifi, X, Zap
} from "lucide-react";
import "./App.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const FEATURE_GROUPS = [
  {
    title: "Connection",
    description: "Protocol and connection state",
    fields: [
      { key: "dur", label: "Duration", type: "number", step: "any", value: 0.001 },
      { key: "proto", label: "Protocol", type: "select", value: "tcp", options: ["tcp", "udp", "unas", "arp", "ospf", "icmp", "sctp", "any", "other"] },
      { key: "service", label: "Service", type: "select", value: "-", options: ["-", "http", "ftp", "ftp-data", "smtp", "ssh", "dns", "pop3", "dhcp", "snmp", "ssl", "radius"] },
      { key: "state", label: "State", type: "select", value: "FIN", options: ["FIN", "INT", "CON", "REQ", "RST", "ACC", "CLO", "ECO", "PAR", "URN", "no"] },
    ],
  },
  {
    title: "Traffic volume",
    description: "Packets, bytes and packet timing",
    fields: [
      { key: "spkts", label: "Source packets", value: 10 },
      { key: "dpkts", label: "Destination packets", value: 8 },
      { key: "sbytes", label: "Source bytes", value: 1200 },
      { key: "dbytes", label: "Destination bytes", value: 980 },
      { key: "rate", label: "Rate", value: 18.2, step: "any" },
      { key: "sttl", label: "Source TTL", value: 64 },
      { key: "dttl", label: "Destination TTL", value: 64 },
      { key: "sload", label: "Source load", value: 9500, step: "any" },
      { key: "dload", label: "Destination load", value: 7600, step: "any" },
      { key: "sloss", label: "Source loss", value: 0 },
      { key: "dloss", label: "Destination loss", value: 0 },
      { key: "sinpkt", label: "Source inter-packet time", value: 0.4, step: "any" },
      { key: "dinpkt", label: "Destination inter-packet time", value: 0.5, step: "any" },
      { key: "sjit", label: "Source jitter", value: 0.01, step: "any" },
      { key: "djit", label: "Destination jitter", value: 0.01, step: "any" },
    ],
  },
  {
    title: "TCP & response",
    description: "Window sizes and response timings",
    fields: [
      { key: "swin", label: "Source window", value: 255 },
      { key: "stcpb", label: "Source TCP base", value: 0 },
      { key: "dtcpb", label: "Destination TCP base", value: 0 },
      { key: "dwin", label: "Destination window", value: 255 },
      { key: "tcprtt", label: "TCP round-trip time", value: 0.02, step: "any" },
      { key: "synack", label: "SYN-ACK time", value: 0.01, step: "any" },
      { key: "ackdat", label: "ACK-data time", value: 0.01, step: "any" },
      { key: "smean", label: "Source mean packet size", value: 120 },
      { key: "dmean", label: "Destination mean packet size", value: 122 },
      { key: "trans_depth", label: "Transaction depth", value: 0 },
      { key: "response_body_len", label: "Response body length", value: 0 },
    ],
  },
  {
    title: "Connection history",
    description: "Recent activity and service indicators",
    fields: [
      { key: "ct_srv_src", label: "Service/source count", value: 2 },
      { key: "ct_state_ttl", label: "State/TTL count", value: 1 },
      { key: "ct_dst_ltm", label: "Destination recent count", value: 1 },
      { key: "ct_src_dport_ltm", label: "Source/destination-port count", value: 1 },
      { key: "ct_dst_sport_ltm", label: "Destination/source-port count", value: 1 },
      { key: "ct_dst_src_ltm", label: "Destination/source recent count", value: 1 },
      { key: "is_ftp_login", label: "FTP login", type: "select", value: 0, options: ["0", "1"] },
      { key: "ct_ftp_cmd", label: "FTP command count", value: 0 },
      { key: "ct_flw_http_mthd", label: "HTTP method count", value: 0 },
      { key: "ct_src_ltm", label: "Source recent count", value: 1 },
      { key: "ct_srv_dst", label: "Service/destination count", value: 1 },
      { key: "is_sm_ips_ports", label: "Same IP/ports", type: "select", value: 0, options: ["0", "1"] },
    ],
  },
];

const INITIAL_FEATURES = Object.fromEntries(
  FEATURE_GROUPS.flatMap((group) => group.fields.map((field) => [field.key, String(field.value ?? "")]))
);

function Field({ field, value, onChange }) {
  return (
    <label className="field">
      <span>{field.label}</span>
      {field.type === "select" ? (
        <select value={value} onChange={(event) => onChange(field.key, event.target.value)}>
          {field.options.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
      ) : (
        <input
          type="number"
          inputMode="decimal"
          step={field.step || "1"}
          value={value}
          onChange={(event) => onChange(field.key, event.target.value)}
          required
        />
      )}
      <small>{field.key}</small>
    </label>
  );
}

function ProbabilityRing({ probability = 0, result = "Awaiting scan" }) {
  const percentage = Math.round(probability * 100);
  return (
    <div className={`probability-ring ${result === "Attack" ? "ring-danger" : "ring-safe"}`} style={{ "--progress": `${percentage}%` }}>
      <div className="ring-inner">
        <span className="ring-value">{percentage}<small>%</small></span>
        <span className="ring-label">attack score</span>
      </div>
    </div>
  );
}

function App() {
  const [features, setFeatures] = useState(INITIAL_FEATURES);
  const [apiState, setApiState] = useState("checking");
  const [apiMessage, setApiMessage] = useState("Checking local inference API");
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
 
  const [mobileNav, setMobileNav] = useState(false);
  const [lastScan, setLastScan] = useState(null);

  const fieldCount = useMemo(() => Object.keys(features).length, [features]);

  async function checkHealth() {
    setApiState("checking");
    setApiMessage("Checking local inference API");

    try {
      const response = await fetch(`${API_BASE}/health`);
      if (!response.ok) {
        throw new Error(`Health check returned ${response.status}`);
      }

      const data = await response.json();
      setApiState("online");
      setApiMessage(`${data.model || "Model"} API connected`);
    } catch {
      setApiState("offline");
      setApiMessage("API unreachable — start FastAPI on port 8000");
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function verifyHealth() {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
          throw new Error(`Health check returned ${response.status}`);
        }

        const data = await response.json();

        if (!cancelled) {
          setApiState("online");
          setApiMessage(`${data.model || "Model"} API connected`);
        }
      } catch {
        if (!cancelled) {
          setApiState("offline");
          setApiMessage("API unreachable — start FastAPI on port 8000");
        }
      }
    }

    verifyHealth();

    return () => {
      cancelled = true;
    };
  }, []);

  function updateFeature(key, value) {
    setFeatures((current) => ({ ...current, [key]: value }));
  }

  async function runScan(event) {
    event.preventDefault();
    setError("");
    setScanning(true);
    setResult(null);

    const numericKeys = new Set(FEATURE_GROUPS.flatMap((group) => group.fields)
      .filter((field) => field.type !== "select" || ["is_ftp_login", "is_sm_ips_ports"].includes(field.key))
      .map((field) => field.key));
    const payload = {};
    for (const [key, value] of Object.entries(features)) {
      if (numericKeys.has(key)) {
        if (value.trim() === "" || !Number.isFinite(Number(value))) {
          setError(`Please enter a valid number for ${key}.`);
          setScanning(false);
          return;
        }
        payload[key] = Number(value);
      } else {
        payload[key] = value;
      }
    }

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ features: payload }),
      });

      const data = await response.json();

      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item) => item.msg).join(", ")
          : data.detail || `Request failed (${response.status})`;
        throw new Error(detail);
      }

      setResult(data);
      setLastScan(new Date());
    } catch (err) {
      setError(err.message || "Could not reach the prediction API.");
    } finally {
      setScanning(false);
      checkHealth();
    }
  }

  function resetFeatures() {
    setFeatures(INITIAL_FEATURES);
    setResult(null);
    setError("");
  }

  const isAttack = result?.prediction === 1;
  const probability = result?.attack_probability ?? 0;

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <header className="topbar">
        <a className="brand" href="#overview" aria-label="CyberSentry AI home">
          <span className="brand-mark"><Shield size={20} strokeWidth={2.3} /></span>
          <span>CyberSentry<span className="brand-ai">.AI</span><small>THREAT INTELLIGENCE</small></span>
        </a>
        <button className="mobile-menu" onClick={() => setMobileNav(!mobileNav)} aria-label="Toggle navigation">
          {mobileNav ? <X size={20} /> : <Menu size={20} />}
        </button>
        <nav className={mobileNav ? "nav-links nav-open" : "nav-links"}>
          <a href="#overview" onClick={() => setMobileNav(false)}>Overview</a>
          <a href="#analysis" onClick={() => setMobileNav(false)}>Traffic analysis</a>
          <a href="#how-it-works" onClick={() => setMobileNav(false)}>How it works</a>
        </nav>
        <div className="topbar-right">
          <span className={`api-pill ${apiState}`}><span className="status-dot" />{apiState === "online" ? "API online" : apiState === "checking" ? "Checking API" : "API offline"}</span>
          <button className="icon-button" title="Recheck API" onClick={checkHealth}><RefreshCw size={16} /></button>
          <a className="top-cta" href="#analysis">Run analysis <ArrowUpRight size={15} /></a>
        </div>
      </header>

      <main>
        <section className="hero" id="overview">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-pulse" /> INTELLIGENCE FOR YOUR NETWORK <span className="eyebrow-line" /></div>
            <h1>See threats<br />before they <span>strike.</span></h1>
            <p className="hero-subtitle">Network intrusion intelligence powered by machine learning. Inspect traffic features, score suspicious activity, and turn raw signals into a clearer security picture.</p>
            <div className="hero-actions">
              <a className="button-primary" href="#analysis"><ScanLine size={17} /> Analyze traffic <ArrowUpRight size={16} /></a>
              <a className="button-secondary" href="#how-it-works">Explore the system <ChevronDown size={16} /></a>
            </div>
            <div className="hero-trust">
              <div className="trust-icon"><ShieldCheck size={17} /></div>
              <div><strong>Random Forest classifier</strong><span>Local inference · UNSW-NB15 feature schema</span></div>
            </div>
          </div>

          <div className="hero-visual" aria-label="Abstract network threat visualization">
            <div className="visual-grid" />
            <div className="orbit orbit-a" /><div className="orbit orbit-b" /><div className="orbit orbit-c" />
            <div className="visual-core">
              <div className="core-halo" />
              <div className="core-shield"><ShieldCheck size={46} strokeWidth={1.3} /></div>
              <span className="core-label">NETWORK DEFENSE</span>
              <span className="core-status"><span /> MONITORING READY</span>
            </div>
            <div className="floating-card card-left">
              <div className="float-card-top"><Activity size={14} /> TRAFFIC FLOW <span className="live-tag">LOCAL</span></div>
              <strong>42 <small>features</small></strong>
              <div className="mini-bars">{Array.from({ length: 18 }, (_, i) => <i key={i} style={{ "--bar-height": `${12 + ((i * 17) % 32)}px` }} />)}</div>
              <span className="float-muted">Structured flow signals</span>
            </div>
            <div className="floating-card card-right">
              <div className="float-card-top"><Fingerprint size={14} /> MODEL ENGINE</div>
              <strong>RF <small>classifier</small></strong>
              <div className="engine-line"><span /><span /><span /><span /><span /><span /><span /></div>
              <span className="float-muted">Probability-based scoring</span>
            </div>
            <div className="visual-bottom"><span><span className="status-dot" /> SYSTEM STANDBY</span><span>CS / 01</span></div>
            <div className="visual-sweep" />
          </div>
        </section>

        <section className="metrics-grid" aria-label="System metrics">
          <article className="metric-card metric-featured">
            <div className="metric-icon"><Radar size={18} /></div>
            <div className="metric-meta"><span>Model input schema</span><span className="metric-tag">READY</span></div>
            <div className="metric-value">{fieldCount}<small> / 42</small></div>
            <div className="metric-foot"><span className="metric-line"><i style={{ width: `${fieldCount / 42 * 100}%` }} /></span><span>features configured</span></div>
          </article>
          <article className="metric-card">
            <div className="metric-icon"><Cpu size={18} /></div>
            <div className="metric-meta"><span>Inference engine</span><span className="metric-tag">ML</span></div>
            <div className="metric-value metric-text">Random Forest</div>
            <div className="metric-foot"><CheckCircle2 size={13} /> Saved pipeline loaded by API</div>
          </article>
          <article className="metric-card">
            <div className="metric-icon"><Gauge size={18} /></div>
            <div className="metric-meta"><span>Decision threshold</span><span className="metric-tag">CONFIG</span></div>
            <div className="metric-value">0.40</div>
            <div className="metric-foot"><span className="threshold-track"><i /></span><span>attack score cutoff</span></div>
          </article>
          <article className="metric-card">
            <div className="metric-icon"><Globe2 size={18} /></div>
            <div className="metric-meta"><span>API connection</span><span className={`tiny-state ${apiState}`} /></div>
            <div className={`metric-value metric-text ${apiState === "online" ? "text-online" : apiState === "offline" ? "text-offline" : ""}`}>{apiState === "online" ? "Connected" : apiState === "checking" ? "Checking…" : "Offline"}</div>
            <div className="metric-foot"><span className={`status-dot ${apiState}`} />{apiMessage}</div>
          </article>
        </section>

        <section className="analysis-section" id="analysis">
          <div className="section-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-pulse" /> ANALYSIS WORKSPACE</div>
              <h2>Inspect a traffic <span>record.</span></h2>
              <p>Enter a single network-flow record using the 42 features expected by your trained model.</p>
            </div>
            <div className="workspace-label"><LockKeyhole size={14} /> LOCAL API · NO FILE UPLOAD</div>
          </div>

          <div className="analysis-layout">
            <form className="feature-panel" onSubmit={runScan}>
              <div className="panel-heading">
                <div className="panel-title-icon"><ScanLine size={17} /></div>
                <div><h3>Traffic parameters</h3><p>UNSW-NB15 compatible feature set</p></div>
                <button type="button" className="text-button" onClick={resetFeatures}>Reset values</button>
              </div>
              <div className="notice"><CircleHelp size={15} /><span>Demo values are prefilled for convenience. Replace them with a real flow record for meaningful analysis.</span></div>
              <div className="feature-groups">
                {FEATURE_GROUPS.map((group, index) => (
                  <details className="feature-group" key={group.title} open={index === 0 || index === 1}>
                    <summary><span className="group-number">0{index + 1}</span><span className="group-summary"><strong>{group.title}</strong><small>{group.description}</small></span><span className="group-count">{group.fields.length} fields</span><ChevronDown className="group-chevron" size={16} /></summary>
                    <div className="field-grid">
                      {group.fields.map((field) => <Field key={field.key} field={field} value={features[field.key]} onChange={updateFeature} />)}
                    </div>
                  </details>
                ))}
              </div>
              {error && <div className="error-banner"><ShieldAlert size={16} /><span>{error}</span></div>}
              <div className="form-actions">
                <span className="form-hint"><LockKeyhole size={13} /> Sent to {API_BASE}</span>
                <button className="button-primary scan-button" type="submit" disabled={scanning}>
                  {scanning ? <><RefreshCw className="spin" size={16} /> Analyzing…</> : <><Zap size={16} /> Analyze record <ArrowUpRight size={16} /></>}
                </button>
              </div>
            </form>

            <aside className="result-column">
              <div className={`result-panel ${result ? (isAttack ? "result-attack" : "result-normal") : ""}`}>
                <div className="result-panel-top"><div className="result-title"><span className="result-icon">{result ? (isAttack ? <Siren size={18} /> : <ShieldCheck size={18} />) : <Activity size={18} />}</span><div><h3>Analysis result</h3><p>{result ? "Model response received" : "Awaiting traffic record"}</p></div></div><span className={`result-badge ${result ? (isAttack ? "badge-danger" : "badge-safe") : "badge-idle"}`}>{result ? (isAttack ? "THREAT FLAGGED" : "NO THREAT FLAGGED") : "IDLE"}</span></div>
                <div className="result-main">
                  <ProbabilityRing probability={probability} result={result?.result || "Awaiting scan"} />
                  <div className="result-description">
                    <span className="overline">{result ? "CLASSIFICATION" : "READY FOR INFERENCE"}</span>
                    <h4>{result ? result.result : "No scan yet"}</h4>
                    <p>{result ? (isAttack ? "The model scored this record above the configured attack threshold. Review the traffic context before taking action." : "The model scored this record below the configured attack threshold. This is not a guarantee that the traffic is safe.") : "Submit the feature form to receive a prediction from your local Random Forest pipeline."}</p>
                  </div>
                </div>
                <div className="result-stats">
                  <div><span>Attack probability</span><strong>{result ? `${(probability * 100).toFixed(2)}%` : "—"}</strong></div>
                  <div><span>Threshold</span><strong>{result ? Number(result.threshold).toFixed(2) : "0.40"}</strong></div>
                  <div><span>Severity label</span><strong className={`severity ${result ? result.severity : "severity-idle"}`}>{result ? result.severity : "—"}</strong></div>
                </div>
                {result && <div className="result-disclaimer"><CircleHelp size={14} /> A model prediction is a signal for review, not proof of malicious activity.</div>}
              </div>

              <div className="activity-panel">
                <div className="activity-heading"><div><h3>Session activity</h3><p>Current browser session</p></div><Clock3 size={16} /></div>
                <div className="activity-row"><span className="activity-icon"><Wifi size={15} /></span><div><strong>API health check</strong><small>{apiMessage}</small></div><span className={`activity-state ${apiState}`}>{apiState === "online" ? "ONLINE" : apiState === "checking" ? "CHECKING" : "OFFLINE"}</span></div>
                <div className="activity-row"><span className="activity-icon"><ScanLine size={15} /></span><div><strong>Last inference</strong><small>{lastScan ? lastScan.toLocaleTimeString() : "No request in this session"}</small></div><span className="activity-state">{lastScan ? "DONE" : "WAITING"}</span></div>
                <div className="activity-row"><span className="activity-icon"><Shield size={15} /></span><div><strong>Data handling</strong><small>Single record sent to local API</small></div><span className="activity-state state-local">LOCAL</span></div>
              </div>
              <div className="help-card"><div className="help-icon"><Sparkles size={17} /></div><div><strong>Designed for investigation</strong><p>Use trusted flow telemetry and investigate flagged records with your wider security tooling.</p><a href="#how-it-works">How scoring works <ArrowUpRight size={13} /></a></div></div>
            </aside>
          </div>
        </section>

        <section className="how-section" id="how-it-works">
          <div className="section-heading compact-heading">
            <div><div className="eyebrow"><span className="eyebrow-pulse" /> UNDER THE HOOD</div><h2>From flow to <span>finding.</span></h2></div>
            <p>One request, a fitted preprocessing pipeline, and a probability-based decision.</p>
          </div>
          <div className="pipeline">
            <article className="pipeline-step"><span className="step-number">01</span><div className="pipeline-icon"><Activity size={20} /></div><h3>Network flow</h3><p>Provide a structured record containing the expected 42 traffic features.</p><span className="pipeline-tag">INPUT</span></article>
            <div className="pipeline-connector"><span /></div>
            <article className="pipeline-step"><span className="step-number">02</span><div className="pipeline-icon"><Cpu size={20} /></div><h3>Preprocessing</h3><p>Numeric scaling and categorical encoding happen inside the saved pipeline.</p><span className="pipeline-tag">TRANSFORM</span></article>
            <div className="pipeline-connector"><span /></div>
            <article className="pipeline-step"><span className="step-number">03</span><div className="pipeline-icon"><Fingerprint size={20} /></div><h3>Random Forest</h3><p>The classifier estimates the probability of the attack class.</p><span className="pipeline-tag">INFERENCE</span></article>
            <div className="pipeline-connector"><span /></div>
            <article className="pipeline-step"><span className="step-number">04</span><div className="pipeline-icon"><ShieldCheck size={20} /></div><h3>Decision</h3><p>The API applies its 0.40 threshold and returns a classification.</p><span className="pipeline-tag">RESPONSE</span></article>
          </div>
        </section>

        <footer className="footer">
          <a className="brand footer-brand" href="#overview"><span className="brand-mark"><Shield size={18} /></span><span>CyberSentry<span className="brand-ai">.AI</span><small>THREAT INTELLIGENCE</small></span></a>
          <span className="footer-note"><span className="status-dot" /> LOCAL INFERENCE INTERFACE</span>
          <span className="footer-copy">Built for exploration · Model outputs require human review</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
