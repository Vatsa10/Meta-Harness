"""envguard: refuse to run real tools against anything that looks like production (stdlib only)."""
import pytest

from meta_harness.integrations.langgraph import envguard
from meta_harness.integrations.langgraph.envguard import ProductionEnvironmentError, check, load_markers


@pytest.mark.parametrize("key,value", [
    ("ENV", "prod"), ("APP_ENV", "Production"), ("NODE_ENV", " production "),
])
def test_env_name_production_blocks(key, value):
    with pytest.raises(ProductionEnvironmentError) as err:
        check({key: value}, [])
    assert err.value.reasons == [f"{key} is set to production"]


@pytest.mark.parametrize("value", ["staging", "dev", "products", "preprod-like", ""])
def test_env_name_non_production_passes(value):
    check({"APP_ENV": value, "NODE_ENV": value}, [])


@pytest.mark.parametrize("key,value", [
    ("ENV_FILE", "config/.env.prod"), ("DOTENV_PATH", "/srv/app/production.env"),
    ("APP_ENV_FILE", "prod.env"),
])
def test_prod_env_file_blocks(key, value):
    with pytest.raises(ProductionEnvironmentError):
        check({key: value}, [])


def test_env_file_only_checks_file_name():
    check({"ENV_FILE": "/home/prod-user/.env.staging"}, [])
    check({"OUTPUT_FILE": "prod.txt"}, [])


def test_markers_match_urls_hosts_and_db_names_without_leaking_values():
    secret = "postgres://admin:hunter2@db.prod.acme.io:5432/orders"
    with pytest.raises(ProductionEnvironmentError) as err:
        check({"DATABASE_URL": secret}, ["DB.PROD.ACME.IO"])
    assert "hunter2" not in str(err.value) and "DATABASE_URL" in str(err.value)
    with pytest.raises(ProductionEnvironmentError):
        check({"REDIS_HOST": "cache-main"}, ["cache-main"])
    with pytest.raises(ProductionEnvironmentError):
        check({"PG_DB_NAME": "orders_main"}, ["orders_main"])
    with pytest.raises(ProductionEnvironmentError):
        check({"WEBHOOK": "https://api.acme.com/hook"}, ["api.acme.com"])


def test_markers_ignore_unrelated_variables_and_other_hosts():
    check({"DATABASE_URL": "postgres://localhost/orders_test", "GREETING": "orders_main"}, ["orders_main"])
    check({"DATABASE_URL": "postgres://db.prod.acme.io/x"}, [])


def test_load_markers_and_default_from_cwd(tmp_path, monkeypatch):
    (tmp_path / "meta-harness.toml").write_text(
        '[production]\nmarkers = ["db.prod.acme.io"]\nhosts = ["10.0.4.12"]\nname = "orders_main"\n',
        encoding="utf-8")
    assert load_markers(tmp_path / "meta-harness.toml") == ["db.prod.acme.io", "10.0.4.12", "orders_main"]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ProductionEnvironmentError):
        check({"API_HOST": "10.0.4.12"})
    check({"API_HOST": "10.0.4.13"})


def test_missing_settings_and_default_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_markers() == []
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(ProductionEnvironmentError):
        check()
    monkeypatch.setenv("APP_ENV", "test")
    for key in envguard.ENV_NAME_KEYS:
        if key != "APP_ENV":
            monkeypatch.delenv(key, raising=False)
    check(production_markers=[])
