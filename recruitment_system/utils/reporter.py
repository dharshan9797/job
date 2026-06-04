"""
Rich terminal report renderer for recruitment pipeline output.
Turns a RecruitmentState into a beautiful, readable terminal report.
"""
from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from models.schemas import (
    HiringDecision,
    InterviewRound,
    RecruitmentState,
)

console = Console()

_DECISION_STYLE = {
    HiringDecision.STRONG_HIRE: ("bold green", "🌟 STRONG HIRE"),
    HiringDecision.HIRE: ("green", "✅ HIRE"),
    HiringDecision.MAYBE: ("yellow", "🤔 MAYBE"),
    HiringDecision.NO_HIRE: ("red", "❌ NO HIRE"),
    HiringDecision.STRONG_NO_HIRE: ("bold red", "🚫 STRONG NO HIRE"),
}

_ROUND_EMOJI = {
    InterviewRound.SCREENING: "📞",
    InterviewRound.TECHNICAL: "💻",
    InterviewRound.BEHAVIORAL: "🧠",
    InterviewRound.SYSTEM_DESIGN: "🏗️",
    InterviewRound.CULTURE_FIT: "🤝",
}


def print_full_report(state: RecruitmentState) -> None:
    """Print the complete recruitment analysis report to the terminal."""

    console.print()
    console.print(Rule("[bold cyan]RECRUITMENT INTELLIGENCE REPORT[/bold cyan]", style="cyan"))
    console.print()

    # --- 1. Candidate Overview ---
    if state.parsed_resume:
        r = state.parsed_resume
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold cyan", width=20)
        overview.add_column()
        overview.add_row("Candidate", r.candidate_name)
        overview.add_row("Email", r.email or "—")
        overview.add_row("Location", r.location or "—")
        overview.add_row("Experience", f"{r.total_experience_years:.1f} years ({r.experience_level.value})")
        overview.add_row("LinkedIn", r.linkedin or "—")
        overview.add_row("GitHub", r.github or "—")
        console.print(Panel(overview, title="[bold]👤 Candidate Profile[/bold]", border_style="cyan"))
        console.print()

    # --- 2. Skill Analysis ---
    if state.skill_analysis:
        sa = state.skill_analysis
        skill_table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta")
        skill_table.add_column("Skill", style="white")
        skill_table.add_column("Category", style="dim")
        skill_table.add_column("Proficiency", justify="center")
        skill_table.add_column("Years", justify="center")

        for sk in sorted(sa.skills, key=lambda x: x.proficiency, reverse=True)[:12]:
            bar_len = int(sk.proficiency * 10)
            bar = f"[green]{'█' * bar_len}[/green][dim]{'░' * (10 - bar_len)}[/dim]"
            console.print()  # spacing
            skill_table.add_row(
                sk.name,
                sk.category.value,
                bar + f" {sk.proficiency:.0%}",
                f"{sk.years_experience:.1f}y",
            )

        console.print(Panel(skill_table, title="[bold]🔧 Skill Analysis[/bold]", border_style="magenta"))

        # Coverage & gaps
        cov_pct = int(sa.coverage_score * 100)
        bar_len = cov_pct // 5
        cov_bar = f"[green]{'█' * bar_len}[/green][dim]{'░' * (20 - bar_len)}[/dim] {cov_pct}%"
        console.print(f"  [bold]Skill Coverage:[/bold] {cov_bar}")

        if sa.skill_gaps:
            gap_table = Table(box=box.SIMPLE, show_header=True, header_style="bold red")
            gap_table.add_column("Missing Skill", style="red")
            gap_table.add_column("Importance", justify="center")
            gap_table.add_column("Gap Severity", justify="center")
            gap_table.add_column("Notes", style="dim")
            for gap in sorted(sa.skill_gaps, key=lambda g: g.importance, reverse=True)[:6]:
                gap_table.add_row(
                    gap.skill,
                    f"{gap.importance:.0%}",
                    f"{gap.gap_severity:.0%}",
                    gap.notes[:60],
                )
            console.print()
            console.print(Panel(gap_table, title="[bold]⚠️  Skill Gaps[/bold]", border_style="red"))

        console.print(f"\n  [italic]{sa.gap_summary}[/italic]")
        console.print()

    # --- 3. Candidate Score ---
    if state.candidate_score:
        sc = state.candidate_score
        score_table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold blue")
        score_table.add_column("Dimension", style="white")
        score_table.add_column("Score", justify="center")
        score_table.add_column("Weight", justify="center")
        score_table.add_column("Rationale", style="dim")

        for dim in sc.dimensions:
            bar_len = int(dim.score)
            bar = f"[{'green' if dim.score >= 7 else 'yellow' if dim.score >= 5 else 'red'}]{'█' * bar_len}[/]{'░' * (10 - bar_len)}"
            score_table.add_row(
                dim.name.replace("_", " ").title(),
                f"{bar} {dim.score:.1f}",
                f"{dim.weight:.0%}",
                dim.rationale[:70],
            )

        overall_style = "green" if sc.overall_score >= 70 else "yellow" if sc.overall_score >= 55 else "red"
        score_table.add_section()
        score_table.add_row(
            "[bold]OVERALL[/bold]",
            f"[bold {overall_style}]{sc.overall_score:.1f}/100[/bold {overall_style}]",
            "",
            sc.scoring_rationale[:70],
        )

        console.print(Panel(score_table, title="[bold]📊 Candidate Scoring[/bold]", border_style="blue"))

        cols = Table.grid(padding=(0, 4), expand=True)
        cols.add_column()
        cols.add_column()

        strengths = "\n".join(f"[green]✓[/green] {s}" for s in sc.strengths)
        concerns = "\n".join(f"[red]✗[/red] {c}" for c in sc.concerns)
        cols.add_row(
            Panel(strengths or "—", title="Strengths", border_style="green", width=45),
            Panel(concerns or "—", title="Concerns", border_style="red", width=45),
        )
        console.print(cols)
        console.print()

    # --- 4. Interview Plan ---
    if state.interview_plan:
        ip = state.interview_plan
        rounds_str = "  →  ".join(
            f"{_ROUND_EMOJI.get(r, '•')} {r.value.replace('_', ' ').title()}"
            for r in ip.recommended_rounds
        )
        console.print(f"  [bold]Interview Rounds:[/bold] {rounds_str}")
        console.print(f"  [bold]Estimated Time:[/bold] {ip.time_estimate_hours}h")
        console.print()

        for rnd in ip.recommended_rounds:
            qs = [q for q in ip.questions if q.round == rnd]
            if not qs:
                continue
            emoji = _ROUND_EMOJI.get(rnd, "•")
            console.print(f"  [bold]{emoji} {rnd.value.replace('_', ' ').title()} Round[/bold]")
            for i, q in enumerate(qs, 1):
                console.print(f"    [cyan]Q{i}:[/cyan] {q.question}")
                console.print(f"         [dim]Purpose: {q.purpose}[/dim]")
                if q.follow_ups:
                    console.print(f"         [dim]Follow-up: {q.follow_ups[0]}[/dim]")
                console.print()

        if ip.red_flags_to_probe:
            flags = "\n".join(f"  ⚑ {f}" for f in ip.red_flags_to_probe)
            console.print(Panel(flags, title="[bold red]🚩 Red Flags to Probe[/bold red]", border_style="red"))
            console.print()

    # --- 5. Final Recommendation ---
    if state.recommendation:
        rec = state.recommendation
        style, label = _DECISION_STYLE.get(rec.decision, ("white", rec.decision.value))
        conf_pct = int(rec.confidence * 100)

        decision_text = Text()
        decision_text.append(f" {label} ", style=f"bold {style} on black")
        decision_text.append(f"  Confidence: {conf_pct}%", style="white")

        console.print(Panel(decision_text, title="[bold]⚖️  HIRING DECISION[/bold]", border_style=style.split()[-1]))
        console.print()
        console.print(f"  [italic]{rec.summary}[/italic]")
        console.print()

        detail_table = Table.grid(padding=(0, 2), expand=False)
        detail_table.add_column(width=30, style="bold")
        detail_table.add_column()

        detail_table.add_row(
            "Key Reasons",
            "\n".join(f"• {r}" for r in rec.key_reasons),
        )
        detail_table.add_row("", "")
        detail_table.add_row(
            "Risk Factors",
            "\n".join(f"• {r}" for r in rec.risk_factors) or "None identified",
        )
        detail_table.add_row("", "")
        detail_table.add_row(
            "Development Plan",
            "\n".join(f"• {d}" for d in rec.development_plan) or "—",
        )
        detail_table.add_row("", "")
        detail_table.add_row("Compensation Band", rec.suggested_compensation_band or "—")
        detail_table.add_row(
            "Start Level",
            rec.suggested_start_level.value if rec.suggested_start_level else "—",
        )
        detail_table.add_row("", "")
        detail_table.add_row(
            "Next Steps",
            "\n".join(f"→ {s}" for s in rec.next_steps),
        )

        console.print(Panel(detail_table, title="[bold]📋 Recommendation Details[/bold]", border_style="white"))

    # --- Errors ---
    if state.errors:
        console.print()
        console.print(Rule("[red]Pipeline Errors[/red]", style="red"))
        for err in state.errors:
            console.print(f"  [red]✗[/red] {err}")

    console.print()
    console.print(Rule("[dim]End of Report[/dim]", style="dim"))
    console.print()
