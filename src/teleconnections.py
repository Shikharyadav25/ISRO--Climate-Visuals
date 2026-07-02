"""
teleconnections.py - NOAA/BOM real-time teleconnection index fetcher.
Fetches ENSO (ONI), IOD, and MJO data from free public APIs.
"""
import json, warnings
from datetime import datetime, timedelta

def fetch_enso_oni():
    """
    Fetch the latest Oceanic Nino Index (ONI) from NOAA CPC.
    Returns dict: {oni_value, phase, trend, impact_india}
    """
    try:
        import urllib.request
        url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            lines = r.read().decode("utf-8", errors="ignore").strip().split("\n")
        # Parse last valid line
        data_lines = [l for l in lines if l.strip() and not l.startswith("SEAS")]
        if not data_lines:
            raise ValueError("No data lines")
        last = data_lines[-1].split()
        oni = float(last[-1])
        if oni >= 2.0:    phase = "Strong El Nino"
        elif oni >= 1.0:  phase = "Moderate El Nino"
        elif oni >= 0.5:  phase = "Weak El Nino"
        elif oni <= -2.0: phase = "Strong La Nina"
        elif oni <= -1.0: phase = "Moderate La Nina"
        elif oni <= -0.5: phase = "Weak La Nina"
        else:             phase = "Neutral"
        impact = _enso_india_impact(phase)
        return {"oni": oni, "phase": phase, "impact_india": impact, "status": "LIVE"}
    except Exception as e:
        return {"oni": None, "phase": "Unknown", "impact_india": "Data unavailable", "status": "ERROR", "error": str(e)}

def _enso_india_impact(phase):
    impacts = {
        "Strong El Nino":     "High risk of below-normal southwest monsoon. Drought conditions likely over central/peninsular India.",
        "Moderate El Nino":   "Moderate risk of deficient monsoon. Watch for July–August rainfall deficits.",
        "Weak El Nino":       "Slight negative bias on monsoon rainfall. Near-normal conditions still possible.",
        "Weak La Nina":       "Slightly favourable for above-normal monsoon over India.",
        "Moderate La Nina":   "Favourable for active monsoon. Enhanced rainfall likely over central and northeast India.",
        "Strong La Nina":     "Very favourable for above-normal monsoon. Flood risk in Ganga-Brahmaputra basin.",
        "Neutral":            "No strong ENSO influence on monsoon. Other factors (IOD, MJO) dominate."
    }
    return impacts.get(phase, "No clear impact signal.")

def fetch_iod():
    """
    Fetch the Dipole Mode Index (DMI/IOD) from NOAA PSL.
    Positive IOD -> enhanced moisture flux into India -> excess monsoon.
    """
    try:
        import urllib.request
        url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            content = r.read().decode("utf-8", errors="ignore")
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        # Last data line has monthly DMI
        data_lines = []
        for l in lines:
            parts = l.split()
            if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 4:
                data_lines.append(parts)
        if not data_lines:
            raise ValueError("No data")
        last = data_lines[-1]
        year = last[0]
        vals = [float(v) for v in last[1:]]
        vals = [v for v in vals if v > -99.0]
        dmi = vals[-1] if vals else 0.0
        if dmi >= 1.0:    phase = "Strong Positive IOD"
        elif dmi >= 0.4:  phase = "Weak Positive IOD"
        elif dmi <= -1.0: phase = "Strong Negative IOD"
        elif dmi <= -0.4: phase = "Weak Negative IOD"
        else:             phase = "Neutral IOD"
        impact = _iod_india_impact(dmi)
        return {"dmi": dmi, "phase": phase, "impact_india": impact, "year": year, "status": "LIVE"}
    except Exception as e:
        return {"dmi": None, "phase": "Unknown", "impact_india": "Data unavailable", "status": "ERROR", "error": str(e)}

def _iod_india_impact(dmi):
    if dmi >= 0.4:
        return "Positive IOD: Enhanced evaporation over western Indian Ocean pushes more moisture into India. Supports above-normal monsoon."
    elif dmi <= -0.4:
        return "Negative IOD: Reduced moisture flux into Indian subcontinent. Can suppress southwest monsoon rainfall."
    return "Neutral IOD: No strong influence on India monsoon from Indian Ocean Dipole."

def fetch_mjo_status():
    """
    Fetch MJO phase from NOAA CPC (RMM index text files).
    Returns current phase (1-8) and amplitude.
    """
    try:
        import urllib.request
        url = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            content = r.read().decode("utf-8", errors="ignore")
        lines = [l for l in content.strip().split("\n") if l.strip() and l.strip()[0].isdigit()]
        if not lines:
            raise ValueError("No data")
        last = lines[-1].split()
        # BOM format: year, month, day, RMM1, RMM2, phase, amplitude
        phase = int(last[5])
        amplitude = float(last[6])
        impact = _mjo_india_impact(phase, amplitude)
        return {"phase": phase, "amplitude": amplitude, "impact_india": impact, "status": "LIVE"}
    except Exception as e:
        return {"phase": None, "amplitude": None, "impact_india": "Data unavailable", "status": "ERROR", "error": str(e)}

MJO_PHASE_DESCRIPTIONS = {
    1: ("Over Africa",          "MJO convection over Africa/Indian Ocean — typically suppressed rainfall over India"),
    2: ("Indian Ocean",         "MJO convection entering western Indian Ocean — rainfall enhancement possible in 5-10 days"),
    3: ("Indian Ocean/Bay",     "Active phase — MJO over Indian Ocean. Enhanced monsoon rainfall likely over India"),
    4: ("Bay of Bengal",        "Active phase — MJO over Bay of Bengal. Strong monsoon enhancement over India/Bangladesh"),
    5: ("Maritime Continent",   "MJO moving toward Maritime Continent — Indian monsoon transitioning to break phase"),
    6: ("Pacific",              "MJO over western Pacific — break/suppressed monsoon conditions likely over India"),
    7: ("Pacific",              "MJO over central Pacific — dry monsoon spell over India"),
    8: ("Western Hemisphere",   "MJO over western hemisphere — weak suppression of Indian monsoon"),
}

def _mjo_india_impact(phase, amplitude):
    if amplitude is None or amplitude < 1.0:
        return "MJO signal is weak (amplitude < 1.0). No significant MJO influence on India monsoon currently."
    desc = MJO_PHASE_DESCRIPTIONS.get(phase, ("Unknown", "No data"))
    return f"Phase {phase} — {desc[0]}. {desc[1]}. Amplitude: {amplitude:.2f} (active if > 1.0)."

def fetch_all_teleconnections():
    """Fetch all teleconnection indices. Returns combined dict."""
    enso = fetch_enso_oni()
    iod  = fetch_iod()
    mjo  = fetch_mjo_status()
    return {"enso": enso, "iod": iod, "mjo": mjo}

