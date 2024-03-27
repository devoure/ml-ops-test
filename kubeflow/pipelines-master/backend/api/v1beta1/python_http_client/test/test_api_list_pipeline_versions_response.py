# coding: utf-8

"""
    Kubeflow Pipelines API

    This file contains REST API specification for Kubeflow Pipelines. The file is autogenerated from the swagger definition.

    Contact: kubeflow-pipelines@google.com
    Generated by: https://openapi-generator.tech
"""


from __future__ import absolute_import

import unittest
import datetime

import kfp_server_api
from kfp_server_api.models.api_list_pipeline_versions_response import ApiListPipelineVersionsResponse  # noqa: E501
from kfp_server_api.rest import ApiException

class TestApiListPipelineVersionsResponse(unittest.TestCase):
    """ApiListPipelineVersionsResponse unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional):
        """Test ApiListPipelineVersionsResponse
            include_option is a boolean, when False only required
            params are included, when True both required and
            optional params are included """
        # model = kfp_server_api.models.api_list_pipeline_versions_response.ApiListPipelineVersionsResponse()  # noqa: E501
        if include_optional :
            return ApiListPipelineVersionsResponse(
                versions = [
                    kfp_server_api.models.api_pipeline_version.apiPipelineVersion(
                        id = '0', 
                        name = '0', 
                        created_at = datetime.datetime.strptime('2013-10-20 19:20:30.00', '%Y-%m-%d %H:%M:%S.%f'), 
                        parameters = [
                            kfp_server_api.models.api_parameter.apiParameter(
                                name = '0', 
                                value = '0', )
                            ], 
                        code_source_url = '0', 
                        package_url = kfp_server_api.models.api_url.apiUrl(
                            pipeline_url = '0', ), 
                        resource_references = [
                            kfp_server_api.models.api_resource_reference.apiResourceReference(
                                key = kfp_server_api.models.api_resource_key.apiResourceKey(
                                    type = 'UNKNOWN_RESOURCE_TYPE', 
                                    id = '0', ), 
                                name = '0', 
                                relationship = 'UNKNOWN_RELATIONSHIP', )
                            ], 
                        description = '0', )
                    ], 
                next_page_token = '0', 
                total_size = 56
            )
        else :
            return ApiListPipelineVersionsResponse(
        )

    def testApiListPipelineVersionsResponse(self):
        """Test ApiListPipelineVersionsResponse"""
        inst_req_only = self.make_instance(include_optional=False)
        inst_req_and_optional = self.make_instance(include_optional=True)


if __name__ == '__main__':
    unittest.main()
