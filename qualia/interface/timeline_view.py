"""
Streamlit interface for the Economic Event Timeline
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from ..events.economic_events import EconomicEvent, EventTimelineManager

def render_timeline(timeline_manager: EventTimelineManager):
    """Render the economic event timeline in Streamlit"""
    st.subheader("🌍 Global Economic Event Timeline")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        impact_filter = st.multiselect(
            "Impact Level",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM"]
        )
    
    with col2:
        category_filter = st.multiselect(
            "Event Category",
            ["MONETARY_POLICY", "ECONOMIC_INDICATOR", "GEOPOLITICAL", 
             "EARNINGS", "MARKET_STRUCTURE", "OTHER"],
            default=["MONETARY_POLICY", "ECONOMIC_INDICATOR"]
        )
    
    with col3:
        time_filter = st.selectbox(
            "Time Range",
            ["Last 24 Hours", "Last Week", "Last Month"],
            index=0
        )

    # Get filtered events
    events = timeline_manager.get_recent_events(limit=100)
    filtered_events = _filter_events(events, impact_filter, category_filter, time_filter)

    if not filtered_events:
        st.info("No events found matching the selected filters.")
        return

    # Display events
    for event in filtered_events:
        _render_event_card(event)

def _filter_events(
    events: List[EconomicEvent],
    impact_filter: List[str],
    category_filter: List[str],
    time_filter: str
) -> List[EconomicEvent]:
    """Filter events based on selected criteria"""
    now = datetime.now()
    time_limits = {
        "Last 24 Hours": now - timedelta(days=1),
        "Last Week": now - timedelta(weeks=1),
        "Last Month": now - timedelta(days=30)
    }
    time_limit = time_limits[time_filter]

    return [
        event for event in events
        if (event.impact.value.upper() in impact_filter and
            event.category.value.upper() in category_filter and
            event.timestamp >= time_limit)
    ]

def _render_event_card(event: EconomicEvent):
    """Render an individual event card"""
    # Calculate time ago
    time_ago = datetime.now() - event.timestamp
    if time_ago < timedelta(hours=24):
        time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
    else:
        time_str = f"{time_ago.days}d ago"

    # Create event card
    with st.container():
        col1, col2 = st.columns([1, 4])
        
        # Impact indicator and timestamp
        with col1:
            impact_colors = {
                "HIGH": "🔴",
                "MEDIUM": "🟡",
                "LOW": "🟢",
                "UNKNOWN": "⚪"
            }
            st.markdown(f"### {impact_colors[event.impact.value.upper()]}")
            st.caption(time_str)

        # Event details
        with col2:
            st.markdown(f"**{event.title}**")
            st.markdown(event.description)
            
            # Metrics row
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric(
                    "Quantum Impact",
                    f"{event.quantum_impact_score:.2f}"
                )
            with metrics_col2:
                st.metric(
                    "Consciousness",
                    f"{event.consciousness_resonance:.2f}"
                )
            with metrics_col3:
                st.metric(
                    "Market Coherence",
                    f"{event.market_coherence_effect:.2f}"
                )

            # Display numerical values if available
            if any(v is not None for v in [event.actual_value, event.forecast_value, event.previous_value]):
                values_df = pd.DataFrame({
                    "Actual": [event.actual_value],
                    "Forecast": [event.forecast_value],
                    "Previous": [event.previous_value]
                }).T
                st.dataframe(values_df)

            # Display related assets
            if event.related_assets:
                st.caption(f"Related Assets: {', '.join(event.related_assets)}")

        st.divider()

def render_impact_summary(timeline_manager: EventTimelineManager):
    """Render a summary of high-impact events and their market effects"""
    st.subheader("📊 Impact Summary")
    
    high_impact_events = timeline_manager.get_high_impact_events()
    if not high_impact_events:
        st.info("No high-impact events currently.")
        return

    # Create summary metrics
    avg_quantum_impact = sum(e.quantum_impact_score for e in high_impact_events) / len(high_impact_events)
    avg_consciousness = sum(e.consciousness_resonance for e in high_impact_events) / len(high_impact_events)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Impact Events", len(high_impact_events))
    with col2:
        st.metric("Avg Quantum Impact", f"{avg_quantum_impact:.2f}")
    with col3:
        st.metric("Avg Consciousness", f"{avg_consciousness:.2f}")
