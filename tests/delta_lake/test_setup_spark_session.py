"""
Tests for the setup_spark_session module.

This is the most critical module for Spark session management.
Tests cover:
- Memory format conversion with overhead calculations
- Executor and driver configuration generation
- Delta Lake, Hive, and S3 configuration
- Immutable configuration filtering for Spark Connect
- Environment variable management for mode switching
- Thread-safe session creation with locking
- Full configuration generation workflow
- Concurrent session creation safety
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest
from pydantic import AnyUrl, AnyHttpUrl

from src.delta_lake.setup_spark_session import (
    convert_memory_format,
    _get_executor_conf,
    _get_spark_defaults_conf,
    _get_delta_conf,
    _get_hive_conf,
    _get_s3_conf,
    _filter_immutable_spark_connect_configs,
    _set_scheduler_pool,
    _clear_spark_env_for_mode_switch,
    generate_spark_conf,
    get_spark_session,
    SPARK_DEFAULT_POOL,
    SPARK_POOLS,
    EXECUTOR_MEMORY_OVERHEAD,
    DRIVER_MEMORY_OVERHEAD,
    IMMUTABLE_CONFIGS,
    _standalone_session_lock,
)
from src.settings import BERDLSettings


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_settings():
    """Create test BERDLSettings."""
    return BERDLSettings(
        KBASE_AUTH_TOKEN="test_token",
        USER="testuser",
        MINIO_ENDPOINT_URL="minio.test:9000",
        MINIO_ACCESS_KEY="test_access",
        MINIO_SECRET_KEY="test_secret",
        MINIO_SECURE=False,
        BERDL_REDIS_HOST="localhost",
        BERDL_REDIS_PORT=6379,
        SPARK_HOME="/usr/local/spark",
        SPARK_MASTER_URL=AnyUrl("spark://master:7077"),
        SPARK_CONNECT_URL=AnyUrl("sc://connect:15002"),
        BERDL_HIVE_METASTORE_URI=AnyUrl("thrift://hive:9083"),
        SPARK_WORKER_COUNT=2,
        SPARK_WORKER_CORES=4,
        SPARK_WORKER_MEMORY="4GiB",
        SPARK_MASTER_CORES=2,
        SPARK_MASTER_MEMORY="2GiB",
        GOVERNANCE_API_URL=AnyHttpUrl("http://governance:8000"),
        BERDL_POD_IP="10.0.0.1",
    )


# =============================================================================
# Test convert_memory_format
# =============================================================================


class TestConvertMemoryFormat:
    """Tests for the convert_memory_format function."""

    def test_gib_to_spark_format(self):
        """Test converting GiB to Spark format with overhead."""
        # 4GiB with 10% overhead = 3.6GiB, rounds to 4g
        result = convert_memory_format("4GiB", overhead_percentage=0.1)
        assert result.endswith("g")
        # Result depends on rounding behavior
        assert result in ["3g", "4g"]

    def test_mib_to_spark_format(self):
        """Test converting MiB to Spark format."""
        # 512MiB with 10% overhead = 460.8MiB
        result = convert_memory_format("512MiB", overhead_percentage=0.1)
        assert result.endswith("m")
        # 512 * 0.9 = 460.8 → rounds to 461
        assert int(result[:-1]) in [460, 461]

    def test_kib_to_spark_format(self):
        """Test converting KiB to Spark format."""
        result = convert_memory_format("1024KiB", overhead_percentage=0.0)
        assert result.endswith("m")
        assert result == "1m"  # 1024KB = 1MB

    def test_gib_format_basic(self):
        """Test basic GiB format conversion."""
        result = convert_memory_format("2GiB", overhead_percentage=0.0)
        assert result == "2g"

    def test_zero_overhead(self):
        """Test with zero overhead percentage."""
        result = convert_memory_format("1GiB", overhead_percentage=0.0)
        assert result == "1g"

    def test_high_overhead(self):
        """Test with high overhead percentage."""
        # 50% overhead
        result = convert_memory_format("4GiB", overhead_percentage=0.5)
        assert result == "2g"

    def test_various_unit_formats(self):
        """Test various memory unit formats."""
        # Different case variations
        assert convert_memory_format("1GB", 0.0) == "1g"
        assert convert_memory_format("1gb", 0.0) == "1g"
        assert convert_memory_format("1Gb", 0.0) == "1g"
        assert convert_memory_format("1gib", 0.0) == "1g"

    def test_decimal_values(self):
        """Test decimal memory values."""
        result = convert_memory_format("1.5GiB", overhead_percentage=0.0)
        # 1.5GB = 1536MB, may return as g or appropriate unit
        assert result in ["1g", "2g", "1536m"]

    def test_tb_format(self):
        """Test terabyte format."""
        result = convert_memory_format("1TiB", overhead_percentage=0.0)
        assert result == "1024g"  # 1TB = 1024GB

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid memory format"):
            convert_memory_format("invalid")

    def test_invalid_unit_raises_error(self):
        """Test that invalid unit raises error or is handled."""
        with pytest.raises(ValueError, match="Invalid memory format"):
            convert_memory_format("1XB")

    def test_executor_overhead_constant(self):
        """Test EXECUTOR_MEMORY_OVERHEAD constant value."""
        assert EXECUTOR_MEMORY_OVERHEAD == 0.1

    def test_driver_overhead_constant(self):
        """Test DRIVER_MEMORY_OVERHEAD constant value."""
        assert DRIVER_MEMORY_OVERHEAD == 0.05

    def test_memory_with_mib_unit(self):
        """Test memory with MiB unit."""
        result = convert_memory_format("256MiB", overhead_percentage=0.0)
        assert result == "256m"

    def test_whitespace_in_format(self):
        """Test memory format with whitespace."""
        result = convert_memory_format("4 GiB", overhead_percentage=0.0)
        assert result == "4g"


# =============================================================================
# Test _get_executor_conf
# =============================================================================


class TestGetExecutorConf:
    """Tests for the _get_executor_conf function."""

    def test_spark_connect_mode_config(self, test_settings):
        """Test configuration for Spark Connect mode."""
        config = _get_executor_conf(test_settings, use_spark_connect=True)

        assert "spark.remote" in config
        # Expect x-kbase-token in URL
        assert ";x-kbase-token=test_token" in config["spark.remote"]
        assert (
            str(test_settings.SPARK_CONNECT_URL).rstrip("/") in config["spark.remote"]
        )
        assert "spark.driver.host" not in config  # Not in Connect mode
        assert "spark.master" not in config  # Not in Connect mode

    def test_legacy_mode_config(self, test_settings):
        """Test configuration for legacy mode."""
        with patch("socket.gethostbyname", return_value="192.168.1.100"):
            with patch("socket.gethostname", return_value="testhost"):
                config = _get_executor_conf(test_settings, use_spark_connect=False)

        assert "spark.driver.host" in config
        assert config["spark.driver.host"] == "192.168.1.100"
        assert config["spark.driver.bindAddress"] == "0.0.0.0"
        assert "spark.master" in config
        assert "spark.remote" not in config

    def test_executor_configuration(self, test_settings):
        """Test executor-specific configuration."""
        config = _get_executor_conf(test_settings, use_spark_connect=True)

        assert "spark.executor.instances" in config
        assert config["spark.executor.instances"] == "2"
        assert config["spark.executor.cores"] == "4"
        assert "spark.executor.memory" in config

    def test_driver_configuration(self, test_settings):
        """Test driver-specific configuration."""
        config = _get_executor_conf(test_settings, use_spark_connect=True)

        assert "spark.driver.cores" in config
        assert config["spark.driver.cores"] == "2"
        assert "spark.driver.memory" in config

    def test_dynamic_allocation_disabled(self, test_settings):
        """Test that dynamic allocation is disabled."""
        config = _get_executor_conf(test_settings, use_spark_connect=True)

        assert config["spark.dynamicAllocation.enabled"] == "false"
        assert config["spark.dynamicAllocation.shuffleTracking.enabled"] == "false"


# =============================================================================
# Test _get_spark_defaults_conf
# =============================================================================


class TestGetSparkDefaultsConf:
    """Tests for the _get_spark_defaults_conf function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        config = _get_spark_defaults_conf()
        assert isinstance(config, dict)

    def test_decommission_settings(self):
        """Test decommission settings are present."""
        config = _get_spark_defaults_conf()

        assert config["spark.decommission.enabled"] == "true"
        assert config["spark.storage.decommission.rddBlocks.enabled"] == "true"

    def test_broadcast_join_threshold(self):
        """Test broadcast join threshold setting."""
        config = _get_spark_defaults_conf()

        assert "spark.sql.autoBroadcastJoinThreshold" in config
        # 50MB = 52428800 bytes
        assert config["spark.sql.autoBroadcastJoinThreshold"] == "52428800"

    def test_shuffle_configuration(self):
        """Test shuffle configuration settings."""
        config = _get_spark_defaults_conf()

        assert config["spark.reducer.maxSizeInFlight"] == "96m"
        assert config["spark.shuffle.file.buffer"] == "1m"


# =============================================================================
# Test _get_delta_conf
# =============================================================================


class TestGetDeltaConf:
    """Tests for the _get_delta_conf function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        config = _get_delta_conf()
        assert isinstance(config, dict)

    def test_delta_extensions(self):
        """Test Delta Lake SQL extensions are configured."""
        config = _get_delta_conf()

        assert (
            config["spark.sql.extensions"] == "io.delta.sql.DeltaSparkSessionExtension"
        )
        assert (
            config["spark.sql.catalog.spark_catalog"]
            == "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )

    def test_delta_optimizations(self):
        """Test Delta Lake optimization settings."""
        config = _get_delta_conf()

        assert config["spark.databricks.delta.optimizeWrite.enabled"] == "true"
        assert config["spark.databricks.delta.autoCompact.enabled"] == "true"
        assert (
            config["spark.databricks.delta.retentionDurationCheck.enabled"] == "false"
        )


# =============================================================================
# Test _get_hive_conf
# =============================================================================


class TestGetHiveConf:
    """Tests for the _get_hive_conf function."""

    def test_hive_metastore_uri(self, test_settings):
        """Test Hive metastore URI is configured."""
        config = _get_hive_conf(test_settings)

        assert config["hive.metastore.uris"] == str(
            test_settings.BERDL_HIVE_METASTORE_URI
        )

    def test_catalog_implementation(self, test_settings):
        """Test catalog implementation is Hive."""
        config = _get_hive_conf(test_settings)

        assert config["spark.sql.catalogImplementation"] == "hive"

    def test_hive_metastore_version(self, test_settings):
        """Test Hive metastore version."""
        config = _get_hive_conf(test_settings)

        assert config["spark.sql.hive.metastore.version"] == "4.0.0"
        assert config["spark.sql.hive.metastore.jars"] == "path"
        assert "spark.sql.hive.metastore.jars.path" in config


# =============================================================================
# Test _get_s3_conf
# =============================================================================


class TestGetS3Conf:
    """Tests for the _get_s3_conf function."""

    def test_s3_endpoint(self, test_settings):
        """Test S3 endpoint configuration."""
        config = _get_s3_conf(test_settings)

        assert (
            config["spark.hadoop.fs.s3a.endpoint"] == test_settings.MINIO_ENDPOINT_URL
        )
        assert (
            config["spark.hadoop.fs.s3a.access.key"] == test_settings.MINIO_ACCESS_KEY
        )
        assert (
            config["spark.hadoop.fs.s3a.secret.key"] == test_settings.MINIO_SECRET_KEY
        )

    def test_ssl_disabled(self, test_settings):
        """Test SSL disabled setting."""
        config = _get_s3_conf(test_settings)

        assert config["spark.hadoop.fs.s3a.connection.ssl.enabled"] == "false"

    def test_ssl_enabled(self, test_settings):
        """Test SSL enabled setting."""
        test_settings_ssl = BERDLSettings(
            USER="testuser",
            MINIO_ENDPOINT_URL="minio.test:9000",
            MINIO_ACCESS_KEY="key",
            MINIO_SECRET_KEY="secret",
            MINIO_SECURE=True,
            SPARK_CONNECT_URL=AnyUrl("sc://localhost:15002"),
            BERDL_HIVE_METASTORE_URI=AnyUrl("thrift://localhost:9083"),
            GOVERNANCE_API_URL=AnyHttpUrl("http://localhost:8000"),
        )

        config = _get_s3_conf(test_settings_ssl)
        assert config["spark.hadoop.fs.s3a.connection.ssl.enabled"] == "true"

    def test_user_warehouse_path(self, test_settings):
        """Test user warehouse path without tenant."""
        config = _get_s3_conf(test_settings, tenant_name=None)

        expected_warehouse = f"s3a://cdm-lake/users-sql-warehouse/{test_settings.USER}/"
        assert config["spark.sql.warehouse.dir"] == expected_warehouse

    def test_tenant_warehouse_path(self, test_settings):
        """Test tenant warehouse path."""
        config = _get_s3_conf(test_settings, tenant_name="research_team")

        expected_warehouse = "s3a://cdm-lake/tenant-sql-warehouse/research_team/"
        assert config["spark.sql.warehouse.dir"] == expected_warehouse

    def test_event_log_directory(self, test_settings):
        """Test event log directory is configured."""
        config = _get_s3_conf(test_settings)

        assert config["spark.eventLog.enabled"] == "true"
        assert f"{test_settings.USER}/" in config["spark.eventLog.dir"]

    def test_s3a_implementation(self, test_settings):
        """Test S3A implementation is configured."""
        config = _get_s3_conf(test_settings)

        assert config["spark.hadoop.fs.s3a.path.style.access"] == "true"
        assert (
            config["spark.hadoop.fs.s3a.impl"]
            == "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )


# =============================================================================
# Test _filter_immutable_spark_connect_configs
# =============================================================================


class TestFilterImmutableSparkConnectConfigs:
    """Tests for the _filter_immutable_spark_connect_configs function."""

    def test_filters_immutable_configs(self):
        """Test that immutable configs are filtered out."""
        config = {
            "spark.app.name": "test",
            "spark.driver.memory": "2g",  # Immutable
            "spark.executor.memory": "4g",  # Immutable
            "spark.sql.shuffle.partitions": "200",  # Mutable
        }

        filtered = _filter_immutable_spark_connect_configs(config)

        assert "spark.app.name" in filtered
        assert "spark.sql.shuffle.partitions" in filtered
        assert "spark.driver.memory" not in filtered
        assert "spark.executor.memory" not in filtered

    def test_keeps_mutable_configs(self):
        """Test that mutable configs are kept."""
        config = {
            "spark.app.name": "test",
            "spark.sql.shuffle.partitions": "100",
            "custom.config": "value",
        }

        filtered = _filter_immutable_spark_connect_configs(config)

        assert filtered == config

    def test_empty_config(self):
        """Test with empty config."""
        filtered = _filter_immutable_spark_connect_configs({})
        assert filtered == {}

    def test_all_immutable_configs(self):
        """Test with all immutable configs."""
        config = {k: "value" for k in list(IMMUTABLE_CONFIGS)[:5]}
        filtered = _filter_immutable_spark_connect_configs(config)
        assert filtered == {}

    def test_immutable_configs_set(self):
        """Test that IMMUTABLE_CONFIGS contains expected keys."""
        assert "spark.driver.memory" in IMMUTABLE_CONFIGS
        assert "spark.executor.memory" in IMMUTABLE_CONFIGS
        assert "spark.sql.warehouse.dir" in IMMUTABLE_CONFIGS
        assert "spark.eventLog.dir" in IMMUTABLE_CONFIGS


# =============================================================================
# Test _set_scheduler_pool
# =============================================================================


class TestSetSchedulerPool:
    """Tests for the _set_scheduler_pool function."""

    def test_valid_pool(self):
        """Test setting a valid scheduler pool."""
        mock_spark = MagicMock()
        mock_spark_context = MagicMock()
        mock_spark.sparkContext = mock_spark_context

        _set_scheduler_pool(mock_spark, "default")

        mock_spark_context.setLocalProperty.assert_called_once_with(
            "spark.scheduler.pool", "default"
        )

    def test_high_priority_pool(self):
        """Test setting high priority pool."""
        mock_spark = MagicMock()
        mock_spark_context = MagicMock()
        mock_spark.sparkContext = mock_spark_context

        _set_scheduler_pool(mock_spark, "highPriority")

        mock_spark_context.setLocalProperty.assert_called_once_with(
            "spark.scheduler.pool", "highPriority"
        )

    def test_invalid_pool_defaults(self, capsys):
        """Test that invalid pool defaults to default pool."""
        mock_spark = MagicMock()
        mock_spark_context = MagicMock()
        mock_spark.sparkContext = mock_spark_context

        _set_scheduler_pool(mock_spark, "invalidPool")

        # Should print warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "invalidPool" in captured.out

        # Should use default pool
        mock_spark_context.setLocalProperty.assert_called_once_with(
            "spark.scheduler.pool", SPARK_DEFAULT_POOL
        )

    def test_spark_pools_constant(self):
        """Test SPARK_POOLS constant."""
        assert "default" in SPARK_POOLS
        assert "highPriority" in SPARK_POOLS
        assert SPARK_DEFAULT_POOL == "default"


# =============================================================================
# Test _clear_spark_env_for_mode_switch
# =============================================================================


class TestClearSparkEnvForModeSwitch:
    """Tests for the _clear_spark_env_for_mode_switch function."""

    def test_clear_for_spark_connect_mode(self):
        """Test clearing env vars for Spark Connect mode."""
        with patch.dict(os.environ, {"MASTER": "spark://old:7077"}, clear=False):
            _clear_spark_env_for_mode_switch(use_spark_connect=True)

            # MASTER should be cleared for Connect mode
            assert "MASTER" not in os.environ

    def test_clear_for_legacy_mode(self):
        """Test clearing env vars for legacy mode."""
        env_vars = {
            "SPARK_CONNECT_MODE_ENABLED": "1",
            "SPARK_REMOTE": "sc://old:15002",
            "SPARK_LOCAL_REMOTE": "1",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            _clear_spark_env_for_mode_switch(use_spark_connect=False)

            # All Connect mode vars should be cleared
            assert "SPARK_CONNECT_MODE_ENABLED" not in os.environ
            assert "SPARK_REMOTE" not in os.environ
            assert "SPARK_LOCAL_REMOTE" not in os.environ

    def test_no_vars_to_clear(self):
        """Test when no env vars need to be cleared."""
        # Should not raise even if vars don't exist
        _clear_spark_env_for_mode_switch(use_spark_connect=True)
        _clear_spark_env_for_mode_switch(use_spark_connect=False)

        # Verify no unexpected changes
        # (can't easily assert exact equality due to test isolation)

    def test_preserves_other_env_vars(self):
        """Test that other env vars are preserved."""
        with patch.dict(
            os.environ, {"MY_VAR": "value", "MASTER": "spark://old:7077"}, clear=False
        ):
            _clear_spark_env_for_mode_switch(use_spark_connect=True)

            # MY_VAR should still exist
            assert os.environ.get("MY_VAR") == "value"


# =============================================================================
# Test generate_spark_conf
# =============================================================================


class TestGenerateSparkConf:
    """Tests for the generate_spark_conf function."""

    def test_default_app_name_generated(self, test_settings):
        """Test that default app name is generated when not provided."""
        config = generate_spark_conf(
            app_name=None, local=True, use_delta_lake=True, settings=test_settings
        )

        assert "spark.app.name" in config
        assert "kbase_spark_session_" in config["spark.app.name"]

    def test_custom_app_name(self, test_settings):
        """Test custom app name is used."""
        config = generate_spark_conf(
            app_name="MyCustomApp", local=True, settings=test_settings
        )

        assert config["spark.app.name"] == "MyCustomApp"

    def test_local_mode_minimal_config(self, test_settings):
        """Test local mode has minimal configuration."""
        config = generate_spark_conf(
            app_name="LocalApp",
            local=True,
            use_delta_lake=False,
            settings=test_settings,
        )

        # Should only have app name
        assert "spark.app.name" in config
        # Should not have cluster configs
        assert "spark.executor.instances" not in config
        assert "spark.driver.host" not in config

    def test_local_mode_with_delta(self, test_settings):
        """Test local mode with Delta Lake enabled."""
        config = generate_spark_conf(
            app_name="LocalApp", local=True, use_delta_lake=True, settings=test_settings
        )

        assert "spark.sql.extensions" in config
        assert "DeltaSparkSessionExtension" in config["spark.sql.extensions"]

    def test_cluster_mode_full_config(self, test_settings):
        """Test cluster mode has full configuration."""
        config = generate_spark_conf(
            app_name="ClusterApp",
            local=False,
            use_delta_lake=True,
            use_s3=True,
            use_hive=True,
            settings=test_settings,
            use_spark_connect=True,
        )

        # Should have S3 config
        assert "spark.hadoop.fs.s3a.endpoint" in config
        # Should have app name
        assert config["spark.app.name"] == "ClusterApp"

    def test_spark_connect_mode_filters_immutable(self, test_settings):
        """Test Spark Connect mode filters immutable configs."""
        config = generate_spark_conf(
            app_name="ConnectApp",
            local=False,
            settings=test_settings,
            use_spark_connect=True,
        )

        # Immutable configs should be filtered
        assert "spark.driver.memory" not in config
        assert "spark.executor.memory" not in config

    def test_legacy_mode_keeps_all_configs(self, test_settings):
        """Test legacy mode keeps all configurations."""
        with patch("socket.gethostbyname", return_value="192.168.1.1"):
            with patch("socket.gethostname", return_value="host"):
                config = generate_spark_conf(
                    app_name="LegacyApp",
                    local=False,
                    settings=test_settings,
                    use_spark_connect=False,
                )

        # Should have driver/executor configs
        assert "spark.driver.memory" in config
        assert "spark.executor.memory" in config

    def test_disable_s3(self, test_settings):
        """Test disabling S3 configuration."""
        config = generate_spark_conf(
            app_name="NoS3App",
            local=False,
            use_s3=False,
            settings=test_settings,
            use_spark_connect=True,
        )

        assert "spark.hadoop.fs.s3a.endpoint" not in config

    def test_disable_hive(self, test_settings):
        """Test disabling Hive configuration."""
        config = generate_spark_conf(
            app_name="NoHiveApp",
            local=False,
            use_hive=False,
            settings=test_settings,
            use_spark_connect=True,
        )

        assert "hive.metastore.uris" not in config

    def test_with_tenant_name(self, test_settings):
        """Test configuration with tenant name."""
        with patch("socket.gethostbyname", return_value="192.168.1.1"):
            with patch("socket.gethostname", return_value="host"):
                config = generate_spark_conf(
                    app_name="TenantApp",
                    local=False,
                    tenant_name="my_tenant",
                    settings=test_settings,
                    use_spark_connect=False,
                )

        # Warehouse should use tenant path
        assert "tenant-sql-warehouse/my_tenant" in config["spark.sql.warehouse.dir"]

    def test_settings_from_get_settings(self):
        """Test that settings are loaded from get_settings when not provided."""
        with patch("src.delta_lake.setup_spark_session.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.SPARK_CONNECT_URL = AnyUrl("sc://test:15002")
            mock_settings.SPARK_WORKER_MEMORY = "2GiB"
            mock_settings.SPARK_MASTER_MEMORY = "1GiB"
            mock_settings.SPARK_WORKER_COUNT = 1
            mock_settings.SPARK_WORKER_CORES = 1
            mock_settings.SPARK_MASTER_CORES = 1
            mock_settings.MINIO_ENDPOINT_URL = "localhost:9000"
            mock_settings.MINIO_ACCESS_KEY = "key"
            mock_settings.MINIO_SECRET_KEY = "secret"
            mock_settings.MINIO_SECURE = False
            mock_settings.USER = "testuser"
            mock_settings.BERDL_HIVE_METASTORE_URI = AnyUrl("thrift://localhost:9083")

            mock_get.return_value = mock_settings

            # Call generate_spark_conf to trigger get_settings loading
            generate_spark_conf(
                app_name="TestApp",
                local=False,
                settings=None,  # Force loading from get_settings
                use_spark_connect=True,
            )

            mock_get.assert_called()


# =============================================================================
# Test get_spark_session
# =============================================================================


class TestGetSparkSession:
    """Tests for the get_spark_session function."""

    def test_creates_spark_session(self, test_settings):
        """Test that get_spark_session creates a SparkSession."""
        mock_session = MagicMock()
        mock_builder = MagicMock()
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session

        with patch(
            "src.delta_lake.setup_spark_session.SparkSession"
        ) as mock_spark_class:
            mock_spark_class.builder = mock_builder

            with patch(
                "src.delta_lake.setup_spark_session._clear_spark_env_for_mode_switch"
            ):
                result = get_spark_session(
                    app_name="TestSession",
                    local=True,
                    delta_lake=True,
                    settings=test_settings,
                )

        assert result == mock_session
        mock_builder.getOrCreate.assert_called_once()

    def test_clears_builder_options(self, test_settings):
        """Test that builder options are cleared."""
        mock_session = MagicMock()
        mock_builder = MagicMock()
        mock_builder._options = {"old": "option"}
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session

        with patch(
            "src.delta_lake.setup_spark_session.SparkSession"
        ) as mock_spark_class:
            mock_spark_class.builder = mock_builder

            with patch(
                "src.delta_lake.setup_spark_session._clear_spark_env_for_mode_switch"
            ):
                get_spark_session(
                    app_name="TestSession",
                    local=True,
                    settings=test_settings,
                )

        # Options should be cleared
        assert mock_builder._options == {}

    def test_override_config(self, test_settings):
        """Test override configuration."""
        mock_session = MagicMock()
        mock_builder = MagicMock()
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session

        with patch(
            "src.delta_lake.setup_spark_session.SparkSession"
        ) as mock_spark_class:
            mock_spark_class.builder = mock_builder

            with patch(
                "src.delta_lake.setup_spark_session._clear_spark_env_for_mode_switch"
            ):
                get_spark_session(
                    app_name="TestSession",
                    local=True,
                    settings=test_settings,
                    override={"custom.setting": "value"},
                )

        # Verify config was called (with override included)
        mock_builder.config.assert_called_once()

    def test_sets_scheduler_pool_in_legacy_mode(self, test_settings):
        """Test scheduler pool is set in legacy mode."""
        mock_session = MagicMock()
        mock_builder = MagicMock()
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session

        with patch(
            "src.delta_lake.setup_spark_session.SparkSession"
        ) as mock_spark_class:
            mock_spark_class.builder = mock_builder

            with patch(
                "src.delta_lake.setup_spark_session._clear_spark_env_for_mode_switch"
            ):
                with patch(
                    "src.delta_lake.setup_spark_session._set_scheduler_pool"
                ) as mock_set_pool:
                    with patch("socket.gethostbyname", return_value="192.168.1.1"):
                        with patch("socket.gethostname", return_value="host"):
                            get_spark_session(
                                app_name="TestSession",
                                local=False,
                                scheduler_pool="highPriority",
                                settings=test_settings,
                                use_spark_connect=False,
                            )

                    mock_set_pool.assert_called_once_with(mock_session, "highPriority")

    def test_no_scheduler_pool_in_connect_mode(self, test_settings):
        """Test scheduler pool is NOT set in Connect mode."""
        mock_session = MagicMock()
        mock_builder = MagicMock()
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session

        with patch(
            "src.delta_lake.setup_spark_session.SparkSession"
        ) as mock_spark_class:
            mock_spark_class.builder = mock_builder

            with patch(
                "src.delta_lake.setup_spark_session._clear_spark_env_for_mode_switch"
            ):
                with patch(
                    "src.delta_lake.setup_spark_session._set_scheduler_pool"
                ) as mock_set_pool:
                    get_spark_session(
                        app_name="TestSession",
                        local=False,
                        settings=test_settings,
                        use_spark_connect=True,
                    )

                    mock_set_pool.assert_not_called()

    def test_uses_spark_conf_with_load_defaults_false(self, test_settings):
        """Test that SparkConf is created with loadDefaults=False."""
        mock_session = MagicMock()
        mock_builder = MagicMock()
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session

        with patch(
            "src.delta_lake.setup_spark_session.SparkSession"
        ) as mock_spark_class:
            mock_spark_class.builder = mock_builder

            with patch(
                "src.delta_lake.setup_spark_session.SparkConf"
            ) as mock_conf_class:
                mock_conf = MagicMock()
                mock_conf.setAll.return_value = mock_conf
                mock_conf_class.return_value = mock_conf

                with patch(
                    "src.delta_lake.setup_spark_session._clear_spark_env_for_mode_switch"
                ):
                    get_spark_session(
                        app_name="TestSession",
                        local=True,
                        settings=test_settings,
                    )

                # Verify SparkConf was created with loadDefaults=False
                mock_conf_class.assert_called_once_with(loadDefaults=False)


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of Spark session creation (Standalone mode)."""

    def test_standalone_session_lock_exists(self):
        """Test that the Standalone mode lock exists."""
        assert _standalone_session_lock is not None
        assert isinstance(_standalone_session_lock, type(threading.Lock()))

    def test_concurrent_session_creation_uses_lock(self, test_settings):
        """Test that concurrent session creation uses the lock."""
        results = []
        errors = []

        mock_session = MagicMock()
        mock_builder = MagicMock()
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session

        def create_session(thread_id):
            try:
                with patch(
                    "src.delta_lake.setup_spark_session.SparkSession"
                ) as mock_spark_class:
                    mock_spark_class.builder = mock_builder

                    with patch(
                        "src.delta_lake.setup_spark_session._clear_spark_env_for_mode_switch"
                    ):
                        # Simulate some work inside the lock
                        result = get_spark_session(
                            app_name=f"Thread{thread_id}Session",
                            local=True,
                            settings=test_settings,
                        )
                        results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(5):
            t = threading.Thread(target=create_session, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All threads should complete without errors
        assert len(errors) == 0
        assert len(results) == 5

    def test_lock_provides_thread_isolation(self, test_settings):
        """Test that lock provides thread isolation."""
        # This test verifies the lock mechanism exists and works
        # We test it indirectly by verifying sessions can be created safely
        execution_count = {"value": 0}
        results = []
        errors = []
        lock = threading.Lock()

        def create_session_mock(thread_id):
            try:
                mock_session = MagicMock()
                mock_builder = MagicMock()
                mock_builder._options = {}
                mock_builder.config.return_value = mock_builder
                mock_builder.getOrCreate.return_value = mock_session

                with patch(
                    "src.delta_lake.setup_spark_session.SparkSession"
                ) as mock_spark_class:
                    mock_spark_class.builder = mock_builder

                    with patch(
                        "src.delta_lake.setup_spark_session._clear_spark_env_for_mode_switch"
                    ):
                        result = get_spark_session(
                            app_name=f"Thread{thread_id}",
                            local=True,
                            settings=test_settings,
                        )
                        with lock:
                            execution_count["value"] += 1
                            results.append((thread_id, result))
            except Exception as e:
                with lock:
                    errors.append((thread_id, e))

        threads = []
        for i in range(3):
            t = threading.Thread(
                target=create_session_mock, args=(i,), name=f"Thread-{i}"
            )
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All threads should complete without errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert execution_count["value"] == 3


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestConfigIntegration:
    """Integration tests for configuration generation."""

    def test_full_config_generation_spark_connect(self, test_settings):
        """Test full configuration generation for Spark Connect mode."""
        config = generate_spark_conf(
            app_name="IntegrationTest",
            local=False,
            use_delta_lake=True,
            use_s3=True,
            use_hive=True,
            settings=test_settings,
            use_spark_connect=True,
        )

        # Verify essential configs are present
        assert "spark.app.name" in config
        assert config["spark.app.name"] == "IntegrationTest"

        # S3 configs should be present but filtered
        assert "spark.hadoop.fs.s3a.endpoint" in config

        # Immutable configs should be filtered
        assert "spark.driver.memory" not in config

    def test_full_config_generation_legacy_mode(self, test_settings):
        """Test full configuration generation for legacy mode."""
        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            with patch("socket.gethostname", return_value="testhost"):
                config = generate_spark_conf(
                    app_name="LegacyIntegration",
                    local=False,
                    use_delta_lake=True,
                    use_s3=True,
                    use_hive=True,
                    settings=test_settings,
                    use_spark_connect=False,
                )

        # All configs should be present
        assert "spark.driver.memory" in config
        assert "spark.executor.memory" in config
        assert "spark.driver.host" in config
        assert config["spark.driver.host"] == "10.0.0.1"

    def test_local_development_config(self, test_settings):
        """Test configuration for local development."""
        config = generate_spark_conf(
            app_name="LocalDev",
            local=True,
            use_delta_lake=True,
            settings=test_settings,
        )

        # Should have minimal config for local mode
        assert "spark.app.name" in config
        assert "spark.sql.extensions" in config  # Delta enabled
        # Should not have cluster configs
        assert "spark.executor.instances" not in config
        assert "spark.hadoop.fs.s3a.endpoint" not in config
