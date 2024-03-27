// Code generated by go-swagger; DO NOT EDIT.

package experiment_service

// This file was generated by the swagger tool.
// Editing this file might prove futile when you re-run the swagger generate command

import (
	"github.com/go-openapi/runtime"

	strfmt "github.com/go-openapi/strfmt"
)

// New creates a new experiment service API client.
func New(transport runtime.ClientTransport, formats strfmt.Registry) *Client {
	return &Client{transport: transport, formats: formats}
}

/*
Client for experiment service API
*/
type Client struct {
	transport runtime.ClientTransport
	formats   strfmt.Registry
}

/*
ExperimentServiceArchiveExperiment archives an experiment and the experiment s runs and recurring runs
*/
func (a *Client) ExperimentServiceArchiveExperiment(params *ExperimentServiceArchiveExperimentParams) (*ExperimentServiceArchiveExperimentOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewExperimentServiceArchiveExperimentParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ExperimentService_ArchiveExperiment",
		Method:             "POST",
		PathPattern:        "/apis/v2beta1/experiments/{experiment_id}:archive",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http"},
		Params:             params,
		Reader:             &ExperimentServiceArchiveExperimentReader{formats: a.formats},
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	return result.(*ExperimentServiceArchiveExperimentOK), nil

}

/*
ExperimentServiceCreateExperiment creates a new experiment
*/
func (a *Client) ExperimentServiceCreateExperiment(params *ExperimentServiceCreateExperimentParams) (*ExperimentServiceCreateExperimentOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewExperimentServiceCreateExperimentParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ExperimentService_CreateExperiment",
		Method:             "POST",
		PathPattern:        "/apis/v2beta1/experiments",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http"},
		Params:             params,
		Reader:             &ExperimentServiceCreateExperimentReader{formats: a.formats},
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	return result.(*ExperimentServiceCreateExperimentOK), nil

}

/*
ExperimentServiceDeleteExperiment deletes an experiment without deleting the experiment s runs and recurring runs to avoid unexpected behaviors delete an experiment s runs and recurring runs before deleting the experiment
*/
func (a *Client) ExperimentServiceDeleteExperiment(params *ExperimentServiceDeleteExperimentParams) (*ExperimentServiceDeleteExperimentOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewExperimentServiceDeleteExperimentParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ExperimentService_DeleteExperiment",
		Method:             "DELETE",
		PathPattern:        "/apis/v2beta1/experiments/{experiment_id}",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http"},
		Params:             params,
		Reader:             &ExperimentServiceDeleteExperimentReader{formats: a.formats},
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	return result.(*ExperimentServiceDeleteExperimentOK), nil

}

/*
ExperimentServiceGetExperiment finds a specific experiment by ID
*/
func (a *Client) ExperimentServiceGetExperiment(params *ExperimentServiceGetExperimentParams) (*ExperimentServiceGetExperimentOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewExperimentServiceGetExperimentParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ExperimentService_GetExperiment",
		Method:             "GET",
		PathPattern:        "/apis/v2beta1/experiments/{experiment_id}",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http"},
		Params:             params,
		Reader:             &ExperimentServiceGetExperimentReader{formats: a.formats},
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	return result.(*ExperimentServiceGetExperimentOK), nil

}

/*
ExperimentServiceListExperiments finds all experiments supports pagination and sorting on certain fields
*/
func (a *Client) ExperimentServiceListExperiments(params *ExperimentServiceListExperimentsParams) (*ExperimentServiceListExperimentsOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewExperimentServiceListExperimentsParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ExperimentService_ListExperiments",
		Method:             "GET",
		PathPattern:        "/apis/v2beta1/experiments",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http"},
		Params:             params,
		Reader:             &ExperimentServiceListExperimentsReader{formats: a.formats},
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	return result.(*ExperimentServiceListExperimentsOK), nil

}

/*
ExperimentServiceUnarchiveExperiment restores an archived experiment the experiment s archived runs and recurring runs will stay archived
*/
func (a *Client) ExperimentServiceUnarchiveExperiment(params *ExperimentServiceUnarchiveExperimentParams) (*ExperimentServiceUnarchiveExperimentOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewExperimentServiceUnarchiveExperimentParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ExperimentService_UnarchiveExperiment",
		Method:             "POST",
		PathPattern:        "/apis/v2beta1/experiments/{experiment_id}:unarchive",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http"},
		Params:             params,
		Reader:             &ExperimentServiceUnarchiveExperimentReader{formats: a.formats},
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	return result.(*ExperimentServiceUnarchiveExperimentOK), nil

}

// SetTransport changes the transport on the client
func (a *Client) SetTransport(transport runtime.ClientTransport) {
	a.transport = transport
}