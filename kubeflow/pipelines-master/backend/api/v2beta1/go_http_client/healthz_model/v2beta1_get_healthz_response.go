// Code generated by go-swagger; DO NOT EDIT.

package healthz_model

// This file was generated by the swagger tool.
// Editing this file might prove futile when you re-run the swagger generate command

import (
	strfmt "github.com/go-openapi/strfmt"

	"github.com/go-openapi/swag"
)

// V2beta1GetHealthzResponse v2beta1 get healthz response
// swagger:model v2beta1GetHealthzResponse
type V2beta1GetHealthzResponse struct {

	// TODO(gkcalat): redesign this service to return status
	// and move server configuration into a separate service
	// TODO(gkcalat): rename or deprecate v1beta1 HealthzService
	//
	// Returns if KFP in multi-user mode
	MultiUser bool `json:"multi_user,omitempty"`
}

// Validate validates this v2beta1 get healthz response
func (m *V2beta1GetHealthzResponse) Validate(formats strfmt.Registry) error {
	return nil
}

// MarshalBinary interface implementation
func (m *V2beta1GetHealthzResponse) MarshalBinary() ([]byte, error) {
	if m == nil {
		return nil, nil
	}
	return swag.WriteJSON(m)
}

// UnmarshalBinary interface implementation
func (m *V2beta1GetHealthzResponse) UnmarshalBinary(b []byte) error {
	var res V2beta1GetHealthzResponse
	if err := swag.ReadJSON(b, &res); err != nil {
		return err
	}
	*m = res
	return nil
}
