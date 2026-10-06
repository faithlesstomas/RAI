"""Profile management; profiles are configuration, conversations are sessions."""

import typer
import yaml

from rai.configuration.storage import load_settings, redact, save_settings

profiles = typer.Typer(
    help="Manage model configuration profiles.", no_args_is_help=True
)


@profiles.command("list")
def list_profiles() -> None:
    settings = load_settings()
    for name in settings.agents:
        typer.echo(f"{name}{' (active)' if name == settings.active_agent else ''}")


@profiles.command("show")
def show_profile(name: str) -> None:
    settings = load_settings()
    if name not in settings.agents:
        raise typer.BadParameter(f"Unknown profile: {name}")
    typer.echo(
        yaml.safe_dump(redact(settings.agents[name].model_dump(exclude_unset=True)))
    )


@profiles.command("use")
def use_profile(name: str) -> None:
    settings = load_settings().runtime_mapping()
    if name not in settings["agents"]:
        raise typer.BadParameter(f"Unknown profile: {name}")
    settings["active_agent"] = name
    save_settings(settings)
    typer.echo(f"Active profile: {name}")


approvals = typer.Typer(help="Manage saved browser search consent for a profile.", no_args_is_help=True)
profiles.add_typer(approvals, name="approvals")


@approvals.command("list")
def list_approvals(profile: str) -> None:
    settings = load_settings()
    if profile not in settings.agents:
        raise typer.BadParameter(f"Unknown profile: {profile}")
    typer.echo(yaml.safe_dump([rule.model_dump() for rule in settings.agents[profile].approval_rules],
                             allow_unicode=True, sort_keys=False))


@approvals.command("set")
def set_approval(  # noqa: PLR0913
    profile: str, rule_id: str, decision: str,
    data_class: str = typer.Option("PUBLIC", "--data-class"),
    query: str | None = typer.Option(None, "--query"),
    all_queries: bool = typer.Option(False, "--all-queries", help="Match any search text, including private data in PRIVATE mode."),
) -> None:
    from pydantic import ValidationError  # noqa: PLC0415
    from rai.configuration.approvals import update_rule  # noqa: PLC0415
    from rai.configuration.models import ApprovalRule  # noqa: PLC0415
    from rai.configuration.storage import ConfigurationError  # noqa: PLC0415

    if (query is None) == (not all_queries):
        raise typer.BadParameter("Specify exactly one of --query TEXT or --all-queries")
    try:
        rule = ApprovalRule(id=rule_id, decision=decision, data_class=data_class, query=query)
        update_rule(profile, rule_id, rule)
    except (ValidationError, ConfigurationError) as exc:
        raise typer.BadParameter("Invalid rule/profile: use allow, ask or deny; class PUBLIC or PRIVATE") from exc
    typer.echo(f"Saved rule {rule_id} for profile {profile}")
    if all_queries and data_class == "PRIVATE" and decision == "allow":
        typer.echo("This authorizes sending ANY search query, including private query text, to search providers.")


@approvals.command("remove")
def remove_approval(profile: str, rule_id: str) -> None:
    from rai.configuration.approvals import update_rule  # noqa: PLC0415
    from rai.configuration.storage import ConfigurationError  # noqa: PLC0415

    try:
        update_rule(profile, rule_id, None)
    except ConfigurationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Removed rule {rule_id} from profile {profile}")
