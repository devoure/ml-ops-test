# kfp-server-api
This file contains REST API specification for Kubeflow Pipelines. The file is autogenerated from the swagger definition.

This Python package is automatically generated by the [OpenAPI Generator](https://openapi-generator.tech) project:

- API version: 2.1.0
- Package version: 2.1.0
- Build package: org.openapitools.codegen.languages.PythonClientCodegen
For more information, please visit [https://www.google.com](https://www.google.com)

## Requirements.

Python 2.7 and 3.4+

## Installation & Usage
### pip install

If the python package is hosted on a repository, you can install directly using:

```sh
pip install git+https://github.com/GIT_USER_ID/GIT_REPO_ID.git
```
(you may need to run `pip` with root permission: `sudo pip install git+https://github.com/GIT_USER_ID/GIT_REPO_ID.git`)

Then import the package:
```python
import kfp_server_api
```

### Setuptools

Install via [Setuptools](http://pypi.python.org/pypi/setuptools).

```sh
python setup.py install --user
```
(or `sudo python setup.py install` to install the package for all users)

Then import the package:
```python
import kfp_server_api
```

## Getting Started

Please follow the [installation procedure](#installation--usage) and then run the following:

```python
from __future__ import print_function

import time
import kfp_server_api
from kfp_server_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = kfp_server_api.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: Bearer
configuration = kfp_server_api.Configuration(
    host = "http://localhost",
    api_key = {
        'authorization': 'YOUR_API_KEY'
    }
)
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['authorization'] = 'Bearer'


# Enter a context with an instance of the API client
with kfp_server_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = kfp_server_api.ExperimentServiceApi(api_client)
    id = 'id_example' # str | The ID of the experiment to be archived.

    try:
        # Archives an experiment and the experiment's runs and jobs.
        api_response = api_instance.experiment_service_archive_experiment_v1(id)
        pprint(api_response)
    except ApiException as e:
        print("Exception when calling ExperimentServiceApi->experiment_service_archive_experiment_v1: %s\n" % e)
    
```

## Documentation for API Endpoints

All URIs are relative to *http://localhost*

Class | Method | HTTP request | Description
------------ | ------------- | ------------- | -------------
*ExperimentServiceApi* | [**experiment_service_archive_experiment_v1**](docs/ExperimentServiceApi.md#experiment_service_archive_experiment_v1) | **POST** /apis/v1beta1/experiments/{id}:archive | Archives an experiment and the experiment&#39;s runs and jobs.
*ExperimentServiceApi* | [**experiment_service_create_experiment_v1**](docs/ExperimentServiceApi.md#experiment_service_create_experiment_v1) | **POST** /apis/v1beta1/experiments | Creates a new experiment.
*ExperimentServiceApi* | [**experiment_service_delete_experiment_v1**](docs/ExperimentServiceApi.md#experiment_service_delete_experiment_v1) | **DELETE** /apis/v1beta1/experiments/{id} | Deletes an experiment without deleting the experiment&#39;s runs and jobs. To avoid unexpected behaviors, delete an experiment&#39;s runs and jobs before deleting the experiment.
*ExperimentServiceApi* | [**experiment_service_get_experiment_v1**](docs/ExperimentServiceApi.md#experiment_service_get_experiment_v1) | **GET** /apis/v1beta1/experiments/{id} | Finds a specific experiment by ID.
*ExperimentServiceApi* | [**experiment_service_list_experiments_v1**](docs/ExperimentServiceApi.md#experiment_service_list_experiments_v1) | **GET** /apis/v1beta1/experiments | Finds all experiments. Supports pagination, and sorting on certain fields.
*ExperimentServiceApi* | [**experiment_service_unarchive_experiment_v1**](docs/ExperimentServiceApi.md#experiment_service_unarchive_experiment_v1) | **POST** /apis/v1beta1/experiments/{id}:unarchive | Restores an archived experiment. The experiment&#39;s archived runs and jobs will stay archived.
*HealthzServiceApi* | [**healthz_service_get_healthz**](docs/HealthzServiceApi.md#healthz_service_get_healthz) | **GET** /apis/v1beta1/healthz | Get healthz data.
*JobServiceApi* | [**job_service_create_job**](docs/JobServiceApi.md#job_service_create_job) | **POST** /apis/v1beta1/jobs | Creates a new job.
*JobServiceApi* | [**job_service_delete_job**](docs/JobServiceApi.md#job_service_delete_job) | **DELETE** /apis/v1beta1/jobs/{id} | Deletes a job.
*JobServiceApi* | [**job_service_disable_job**](docs/JobServiceApi.md#job_service_disable_job) | **POST** /apis/v1beta1/jobs/{id}/disable | Stops a job and all its associated runs. The job is not deleted.
*JobServiceApi* | [**job_service_enable_job**](docs/JobServiceApi.md#job_service_enable_job) | **POST** /apis/v1beta1/jobs/{id}/enable | Restarts a job that was previously stopped. All runs associated with the job will continue.
*JobServiceApi* | [**job_service_get_job**](docs/JobServiceApi.md#job_service_get_job) | **GET** /apis/v1beta1/jobs/{id} | Finds a specific job by ID.
*JobServiceApi* | [**job_service_list_jobs**](docs/JobServiceApi.md#job_service_list_jobs) | **GET** /apis/v1beta1/jobs | Finds all jobs.
*PipelineServiceApi* | [**pipeline_service_create_pipeline_v1**](docs/PipelineServiceApi.md#pipeline_service_create_pipeline_v1) | **POST** /apis/v1beta1/pipelines | Creates a pipeline.
*PipelineServiceApi* | [**pipeline_service_create_pipeline_version_v1**](docs/PipelineServiceApi.md#pipeline_service_create_pipeline_version_v1) | **POST** /apis/v1beta1/pipeline_versions | Adds a pipeline version to the specified pipeline.
*PipelineServiceApi* | [**pipeline_service_delete_pipeline_v1**](docs/PipelineServiceApi.md#pipeline_service_delete_pipeline_v1) | **DELETE** /apis/v1beta1/pipelines/{id} | Deletes a pipeline and its pipeline versions.
*PipelineServiceApi* | [**pipeline_service_delete_pipeline_version_v1**](docs/PipelineServiceApi.md#pipeline_service_delete_pipeline_version_v1) | **DELETE** /apis/v1beta1/pipeline_versions/{version_id} | Deletes a pipeline version by pipeline version ID. If the deleted pipeline version is the default pipeline version, the pipeline&#39;s default version changes to the pipeline&#39;s most recent pipeline version. If there are no remaining pipeline versions, the pipeline will have no default version. Examines the run_service_api.ipynb notebook to learn more about creating a run using a pipeline version (https://github.com/kubeflow/pipelines/blob/master/tools/benchmarks/run_service_api.ipynb).
*PipelineServiceApi* | [**pipeline_service_get_pipeline_by_name_v1**](docs/PipelineServiceApi.md#pipeline_service_get_pipeline_by_name_v1) | **GET** /apis/v1beta1/namespaces/{namespace}/pipelines/{name} | Finds a pipeline by Name (and namespace)
*PipelineServiceApi* | [**pipeline_service_get_pipeline_v1**](docs/PipelineServiceApi.md#pipeline_service_get_pipeline_v1) | **GET** /apis/v1beta1/pipelines/{id} | Finds a specific pipeline by ID.
*PipelineServiceApi* | [**pipeline_service_get_pipeline_version_template**](docs/PipelineServiceApi.md#pipeline_service_get_pipeline_version_template) | **GET** /apis/v1beta1/pipeline_versions/{version_id}/templates | Returns a YAML template that contains the specified pipeline version&#39;s description, parameters and metadata.
*PipelineServiceApi* | [**pipeline_service_get_pipeline_version_v1**](docs/PipelineServiceApi.md#pipeline_service_get_pipeline_version_v1) | **GET** /apis/v1beta1/pipeline_versions/{version_id} | Gets a pipeline version by pipeline version ID.
*PipelineServiceApi* | [**pipeline_service_get_template**](docs/PipelineServiceApi.md#pipeline_service_get_template) | **GET** /apis/v1beta1/pipelines/{id}/templates | Returns a single YAML template that contains the description, parameters, and metadata associated with the pipeline provided.
*PipelineServiceApi* | [**pipeline_service_list_pipeline_versions_v1**](docs/PipelineServiceApi.md#pipeline_service_list_pipeline_versions_v1) | **GET** /apis/v1beta1/pipeline_versions | Lists all pipeline versions of a given pipeline.
*PipelineServiceApi* | [**pipeline_service_list_pipelines_v1**](docs/PipelineServiceApi.md#pipeline_service_list_pipelines_v1) | **GET** /apis/v1beta1/pipelines | Finds all pipelines.
*PipelineServiceApi* | [**pipeline_service_update_pipeline_default_version_v1**](docs/PipelineServiceApi.md#pipeline_service_update_pipeline_default_version_v1) | **POST** /apis/v1beta1/pipelines/{pipeline_id}/default_version/{version_id} | Update the default pipeline version of a specific pipeline.
*PipelineUploadServiceApi* | [**upload_pipeline**](docs/PipelineUploadServiceApi.md#upload_pipeline) | **POST** /apis/v1beta1/pipelines/upload | 
*PipelineUploadServiceApi* | [**upload_pipeline_version**](docs/PipelineUploadServiceApi.md#upload_pipeline_version) | **POST** /apis/v1beta1/pipelines/upload_version | 
*RunServiceApi* | [**run_service_archive_run_v1**](docs/RunServiceApi.md#run_service_archive_run_v1) | **POST** /apis/v1beta1/runs/{id}:archive | Archives a run.
*RunServiceApi* | [**run_service_create_run_v1**](docs/RunServiceApi.md#run_service_create_run_v1) | **POST** /apis/v1beta1/runs | Creates a new run.
*RunServiceApi* | [**run_service_delete_run_v1**](docs/RunServiceApi.md#run_service_delete_run_v1) | **DELETE** /apis/v1beta1/runs/{id} | Deletes a run.
*RunServiceApi* | [**run_service_get_run_v1**](docs/RunServiceApi.md#run_service_get_run_v1) | **GET** /apis/v1beta1/runs/{run_id} | Finds a specific run by ID.
*RunServiceApi* | [**run_service_list_runs_v1**](docs/RunServiceApi.md#run_service_list_runs_v1) | **GET** /apis/v1beta1/runs | Finds all runs.
*RunServiceApi* | [**run_service_read_artifact_v1**](docs/RunServiceApi.md#run_service_read_artifact_v1) | **GET** /apis/v1beta1/runs/{run_id}/nodes/{node_id}/artifacts/{artifact_name}:read | Finds a run&#39;s artifact data.
*RunServiceApi* | [**run_service_report_run_metrics_v1**](docs/RunServiceApi.md#run_service_report_run_metrics_v1) | **POST** /apis/v1beta1/runs/{run_id}:reportMetrics | ReportRunMetrics reports metrics of a run. Each metric is reported in its own transaction, so this API accepts partial failures. Metric can be uniquely identified by (run_id, node_id, name). Duplicate reporting will be ignored by the API. First reporting wins.
*RunServiceApi* | [**run_service_retry_run_v1**](docs/RunServiceApi.md#run_service_retry_run_v1) | **POST** /apis/v1beta1/runs/{run_id}/retry | Re-initiates a failed or terminated run.
*RunServiceApi* | [**run_service_terminate_run_v1**](docs/RunServiceApi.md#run_service_terminate_run_v1) | **POST** /apis/v1beta1/runs/{run_id}/terminate | Terminates an active run.
*RunServiceApi* | [**run_service_unarchive_run_v1**](docs/RunServiceApi.md#run_service_unarchive_run_v1) | **POST** /apis/v1beta1/runs/{id}:unarchive | Restores an archived run.


## Documentation For Models

 - [ApiCronSchedule](docs/ApiCronSchedule.md)
 - [ApiExperiment](docs/ApiExperiment.md)
 - [ApiExperimentStorageState](docs/ApiExperimentStorageState.md)
 - [ApiGetHealthzResponse](docs/ApiGetHealthzResponse.md)
 - [ApiGetTemplateResponse](docs/ApiGetTemplateResponse.md)
 - [ApiJob](docs/ApiJob.md)
 - [ApiListExperimentsResponse](docs/ApiListExperimentsResponse.md)
 - [ApiListJobsResponse](docs/ApiListJobsResponse.md)
 - [ApiListPipelineVersionsResponse](docs/ApiListPipelineVersionsResponse.md)
 - [ApiListPipelinesResponse](docs/ApiListPipelinesResponse.md)
 - [ApiListRunsResponse](docs/ApiListRunsResponse.md)
 - [ApiParameter](docs/ApiParameter.md)
 - [ApiPeriodicSchedule](docs/ApiPeriodicSchedule.md)
 - [ApiPipeline](docs/ApiPipeline.md)
 - [ApiPipelineRuntime](docs/ApiPipelineRuntime.md)
 - [ApiPipelineSpec](docs/ApiPipelineSpec.md)
 - [ApiPipelineVersion](docs/ApiPipelineVersion.md)
 - [ApiReadArtifactResponse](docs/ApiReadArtifactResponse.md)
 - [ApiRelationship](docs/ApiRelationship.md)
 - [ApiReportRunMetricsRequest](docs/ApiReportRunMetricsRequest.md)
 - [ApiReportRunMetricsResponse](docs/ApiReportRunMetricsResponse.md)
 - [ApiResourceKey](docs/ApiResourceKey.md)
 - [ApiResourceReference](docs/ApiResourceReference.md)
 - [ApiResourceType](docs/ApiResourceType.md)
 - [ApiRun](docs/ApiRun.md)
 - [ApiRunDetail](docs/ApiRunDetail.md)
 - [ApiRunMetric](docs/ApiRunMetric.md)
 - [ApiRunStorageState](docs/ApiRunStorageState.md)
 - [ApiStatus](docs/ApiStatus.md)
 - [ApiTrigger](docs/ApiTrigger.md)
 - [ApiUrl](docs/ApiUrl.md)
 - [GatewayruntimeError](docs/GatewayruntimeError.md)
 - [JobMode](docs/JobMode.md)
 - [PipelineSpecRuntimeConfig](docs/PipelineSpecRuntimeConfig.md)
 - [ProtobufAny](docs/ProtobufAny.md)
 - [ProtobufNullValue](docs/ProtobufNullValue.md)
 - [ReportRunMetricsResponseReportRunMetricResult](docs/ReportRunMetricsResponseReportRunMetricResult.md)
 - [ReportRunMetricsResponseReportRunMetricResultStatus](docs/ReportRunMetricsResponseReportRunMetricResultStatus.md)
 - [RunMetricFormat](docs/RunMetricFormat.md)


## Documentation For Authorization


## Bearer

- **Type**: API key
- **API key parameter name**: authorization
- **Location**: HTTP header


## Author

kubeflow-pipelines@google.com

