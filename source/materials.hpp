// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for material definitions.
*/

#ifndef MATERIALS_H
#define MATERIALS_H

#include <stdexcept>

#include "types.hpp"
#include "constants.hpp"


namespace bem
{

const Float CONDUCTOR_LOSS_TAN_THRESHOLD = 10;

/**
* \defgroup mater Materials
* @brief Classes for defining and managing materials.
* @{
*/

/**
* @brief Class defining a general material with a constant (zero or non-zero) electrical conductivity
* and real or complex permittivity and permeability.
*/
class Material
{
public:

    /**
    * @brief Constructs a `Material` with given relative permittivity, relative permeability,
    * and electrical conductivity.
    * @param[in] epsr - Relative permittivity of the material, can be complex (optional).
    * @param[in] mur - Relative permeability of the material, can be complex (optional).
    * @param[in] sigma - Electrical conductivity of the material (optional).
    */
    Material(
        const Complex epsr = one,
        const Complex mur = one,
        const Float sigma = 0
        ): epsr_(epsr), mur_(mur), sigma_(sigma) {};


    /**
    * @brief Returns the permittivity.
    * @return Dielectric permittivity of the material.
    */
    virtual Complex eps() const { return epsr_ * eps0; };


    /**
    * @brief Returns the relative permittivity.
    * @return Relative dielectric permittivity of the material.
    */
    virtual Complex epsr() const { return epsr_; };


    /**
    * @brief Returns the effective complex permittivity taking into account the electric conductivity.
    * @return Effective complex permittivity of the material, taking into account the conductivity.
    */
    virtual Complex eps_eff(const Float f) const
    {
        if (std::abs(f) <= float_eps)
            return eps();
        else
            return eps() - J * sigma() / f / two_pi;
    };


    /**
    * @brief Returns the relative effective complex permittivity taking into account the electric conductivity.
    * @return Relative effective complex permittivity of the material, taking into account the conductivity.
    */
    virtual Complex epsr_eff(const Float f) const { return eps_eff(f) / eps0; };


    /**
    * @brief Returns the permeability.
    * @return Magnetic permeability of the material.
    */
    virtual Complex mu() const { return mur_ * mu0; };


    /**
    * @brief Returns the relative permeability.
    * @return Relative magnetic permeability of the material.
    */
    virtual Complex mur() const { return mur_; };


    /**
    * @brief Returns the electrical conductivity.
    * @return Electrical conductivity of the material.
    */
    virtual Float sigma() const { return sigma_; };


    /**
    * @brief Returns the frequency-dependent loss tangent.
    * @param[in] f - Frequency at which to compute the loss tangent.
    * @return Loss tangent of the material.
    */
    virtual Float loss_tan(const Float f) const
    { return -std::imag(eps_eff(f)) / std::real(eps_eff(f)); };


    /**
    * @brief Returns the frequency-dependent complex wave impedance.
    * @param[in] f - Frequency at which to compute the wave impedance.
    * @return Complex wave impedance associated with the material.
    */
    virtual Complex eta(const Float f) const
    {
        if (std::abs(sigma()) <= float_eps)
            return std::sqrt(mu() / eps());
        else
            return std::sqrt((J * two_pi * f * mu()) / (J * two_pi * f * eps() + sigma()));
    }
    // { return std::sqrt(mu() / eps_eff(f)); };


    /**
    * @brief Returns the frequency-dependent complex wavenumber.
    * @param[in] f - Frequency at which to compute the wavenumber.
    * @return Complex wavenumber associated with the material.
    */
    virtual Complex k(const Float f) const
    { return std::sqrt((-J * two_pi * f * mu()) * (J * two_pi * f * eps() + sigma())); };
    // { return two_pi * f * std::sqrt(eps_eff(f) * mu()); };


    /**
    * @brief Returns the frequency-dependent wavelength in the material.
    * @param[in] f - Frequency at which to compute the wavelength.
    * @return Wavelength associated with the material.
    */
    virtual Float wvl(const Float f) const
    { return std::real(two_pi / k(f)); };


    /**
    * @brief Returns the frequency-dependent wave velocity in the material.
    * @param[in] f - Frequency at which to compute the wave velocity.
    * @return Wave velocity associated with the material.
    */
    virtual Complex c(const Float f) const
    { return one / std::sqrt(eps_eff(f) * mu()); };


    /**
    * @brief Returns the frequency-dependent skin depth in the material.
    * @param[in] f - Frequency at which to compute the skin depth.
    * @return Skin depth associated with the material.
    */
    virtual Float skin_depth(const Float f) const
    { return -one / std::imag(k(f)); };


    /**
    * @brief Checks whether the material can be treated as a good conductor at a given frequency.
    * @param[in] f - Frequency to check.
    * @return `true` if the material can be treated as a good conductor, `false` otherwise.
    */
    virtual bool good_conductor(const Float f) const
    { return loss_tan(f) > CONDUCTOR_LOSS_TAN_THRESHOLD; };


    /**
    * @brief Virtual destructor.
    */
    virtual ~Material() = default;


protected:

    const Complex epsr_ = one;
    const Complex mur_ = one;
    const Float sigma_ = 0;

};


/**
* @brief Class defining a perfect lossless dielectric material.
*/
class PerfectDielectricMaterial: public Material
{

    using Material::eta;
    using Material::c;

public:

    /**
    * @brief Constructs a `PerfectDielectricMaterial` with given relative permittivity and permeability.
    * @param[in] epsr - Relative permittivity of the material (optional).
    * @param[in] mur - Relative permeability of the material (optional).
    */
    PerfectDielectricMaterial(
        const Float epsr = one,
        const Float mur = one
        ): Material(epsr, mur, 0) {};


    /**
    * @brief Returns the wave impedance.
    * @return Wave impedance associated with the material.
    */
    Float eta() const
    { return std::real(std::sqrt(mu() / eps())); };


    /**
    * @brief Returns the wave velocity.
    * @return Wave velocity associated with the material.
    */
    Float c() const
    { return one / std::real(std::sqrt(mu() * eps())); };

};


/**
* @brief Class defining a material with a constant loss tangent and real permittivity and permeability.
*/
class ConstantLossTangentMaterial: public Material
{
public:

    /**
    * @brief Constructs a `ConstantLossTangentMaterial` with given relative permittivity,
    * relative permeability, and loss tangent.
    * @param[in] epsr - Relative permittivity of the material (optional).
    * @param[in] mur - Relative permeability of the material (optional).
    * @param[in] loss_tan - Loss tangent of the material (optional).
    */
    ConstantLossTangentMaterial(
        const Float epsr = one,
        const Float mur = one,
        const Float loss_tan = 0
        ): Material(epsr, mur, 0), loss_tan_(loss_tan) {};


    /**
    * @brief Returns the complex permittivity.
    * @return Complex dielectric permittivity of the material.
    */
    Complex eps() const override
    { return epsr_ * eps0 * (one - J * loss_tan_); };


    /**
    * @brief Returns the complex relative permittivity.
    * @return Complex relative dielectric permittivity of the material.
    */
    Complex epsr() const override
    { return eps() / eps0; };


    /**
    * @brief Returns the complex permittivity.
    * @return Complex dielectric permittivity of the material.
    */
    Complex eps_eff(const Float f) const override
    { return eps(); };


    /**
    * @brief Returns the loss tangent.
    * @return Loss tangent of the material.
    */
    Float loss_tan() const
    { return loss_tan_; };


    /**
    * @brief Returns the loss tangent.
    * @param[in] f - Frequency at which to compute the loss tangent - not used in this case since
    * the loss tangent is constant by definition.
    * @return Loss tangent of the material.
    */
    Float loss_tan(const Float f) const override
    { return loss_tan(); };


protected:

    const Float loss_tan_ = 0;

};

/**
* @}
*/

}

#endif
