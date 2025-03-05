//  MIT License
//  
//  Copyright (c) 2025 Severi Suominen
//  
//  Permission is hereby granted, free of charge, to use, copy, modify, merge,
//  publish, distribute, sublicense, and/or sell copies of this software, 
//  provided that the above copyright notice and this permission notice appear 
//  in all copies or substantial portions of the software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
//  FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
//
//  GitHub: https://github.com/SeveriSuominen

#pragma once 

#include <cmath>
#include <cstdint>
#include <cassert>
#include <limits>
#include <type_traits>

#if defined (__clang__)
#	define MXLIB_NODISCARD [[nodiscard]]
#else
#	define MXLIB_NODISCARD
#endif
#define MXLIB_INLINE inline
#define MXLIB_FUNC_DECL MXLIB_NODISCARD MXLIB_INLINE
#define MXLIB_CONSTEXPR MXLIB_INLINE static constexpr
#define MXLIB_STATIC_STRUCT(s) s() = delete;

namespace mxlib
{
    ////////////////////////
    // constant
    ///////////////////////

    template<typename T, typename = std::enable_if<std::is_arithmetic_v<T>>>
    struct spec 
    { 
        MXLIB_CONSTEXPR T min = std::numeric_limits<T>::min();

        MXLIB_CONSTEXPR T max = std::numeric_limits<T>::max();
        
        MXLIB_STATIC_STRUCT(spec);
    };

    template<>
    struct spec<float>
    { 
        MXLIB_CONSTEXPR float pi = 3.141593f;
        
        MXLIB_CONSTEXPR float epsilon = 0.00001f;

        MXLIB_CONSTEXPR float epsilon_norm_sq = 1e-15f;

        MXLIB_CONSTEXPR float deg_to_rad = 0.01745329f;

        MXLIB_CONSTEXPR float rad_to_deg = 57.29578f;

        MXLIB_CONSTEXPR float inf  = std::numeric_limits<float>::infinity();

        MXLIB_CONSTEXPR float ninf = -std::numeric_limits<float>::infinity();

        MXLIB_STATIC_STRUCT(spec);
    };

    template<>
    struct spec<double>
    { 
        MXLIB_CONSTEXPR double pi = 3.14159265358979323846;

        MXLIB_CONSTEXPR double epsilon = 0.0000000000001;

        MXLIB_CONSTEXPR double epsilon_norm_sq = 1e-30;

        MXLIB_CONSTEXPR double deg_to_rad = 0.017453292519943295;

        MXLIB_CONSTEXPR double rad_to_deg = 57.29577951308232;

        MXLIB_CONSTEXPR double inf  = std::numeric_limits<double>::infinity();

        MXLIB_CONSTEXPR double ninf = -std::numeric_limits<double>::infinity();

        MXLIB_STATIC_STRUCT(spec);
    };

    ////////////////////////
    // utility
    ///////////////////////

    template<typename T, typename U, typename = std::enable_if_t<sizeof(T) == sizeof(U)>>
    MXLIB_FUNC_DECL static T reinterpret(U&& src_) {
        constexpr auto valid_types = 
            std::is_pointer_v<T> == std::is_pointer_v<U>;
        static_assert(valid_types, "invalid type combination");
        if constexpr (std::is_pointer_v<T>){
            return reinterpret_cast<T>(src_);
        };
        T dst_;
        std::memcpy(&dst_, &src_, sizeof(T));
        return dst_;
    }

    ////////////////////////
    // math structures
    ///////////////////////

    template<typename T, uint32_t D,
    typename = std::enable_if<std::is_arithmetic_v<T>>>
    struct vector_t 
    {
        using value_type = T;

        vector_t() = default;

        vector_t(const vector_t&) = default;

        vector_t(vector_t&&) = default;
        
        vector_t& operator=(const vector_t&) = default;
        
        vector_t& operator=(vector_t&&) = default;

        template<typename... ARGS, 
        typename = std::enable_if_t<(sizeof...(ARGS)<=D) 
        && (std::conjunction_v<std::is_convertible<ARGS, T>...>)>>
        vector_t(ARGS&&... args): _v(args...) {}

        explicit vector_t(T* array_) {
            std::memcpy(data(), array_, sizeof(T)*D);
        }

        template<typename... ARGS>
        vector_t<T, sizeof...(ARGS)> swizzle(ARGS&&... idxs) const {
            constexpr auto args_size = sizeof...(ARGS); 
            static_assert(args_size <= D, "overflow");
            T args_[args_size] {idxs...}; 

            vector_t<T, sizeof...(ARGS)> vec_{};
            for(int i = 0; i < args_size; ++i){
                vec_[i] = _v[args_[i]];
            }
            return vec_;
        }
   
        T&       operator[](uint32_t i) { return _v[i];}

        const T& operator[](uint32_t i) const { return _v[i];}

        vector_t operator*(T scalar_) const { return {_v[0]*scalar_, _v[1]*scalar_, _v[2]*scalar_}; }

        vector_t operator/(T scalar_) const { return {_v[0]/scalar_, _v[1]/scalar_, _v[2]/scalar_}; }

        vector_t operator+(vector_t other_) const { return {_v[0]+other_[0], _v[1]+other_[1], _v[2]+other_[2]}; }

        vector_t operator-(vector_t other_) const { return {_v[0]-other_[0], _v[1]-other_[1], _v[2]-other_[2]}; }

        T* data() { return &_v[0]; }

        const T* data() const { return &_v[0]; }

    private:
        T _v[D]{0};
    };

    template<typename T, uint32_t D0, uint32_t D1, 
    typename = std::enable_if<std::is_arithmetic_v<T>>>
    struct matrix_t 
    {
        matrix_t() = default;

        matrix_t(const matrix_t&) = default;

        matrix_t(matrix_t&&) = default;
        
        matrix_t& operator=(const matrix_t&) = default;
        
        matrix_t& operator=(matrix_t&&) = default;
        
        explicit matrix_t(T *array_) {
          std::memcpy(data(), array_, sizeof(T) * D0 * D1);
        }

        template<typename... ARGS, 
        typename = std::enable_if_t<(sizeof...(ARGS)<=D0*D1) 
        && (std::conjunction_v<std::is_convertible<ARGS, T>...>)>>
        matrix_t(ARGS&&... args): _m(args...) {}

        T&       operator[](uint32_t i) { return _m[i];}

        const T& operator[](uint32_t i) const { return _m[i];}

        T& val(uint32_t x, uint32_t y) { return _m[y * D0 + x];}

        const T& val(uint32_t x, uint32_t y) const { return _m[y * D0 + x];}

        T* data() { return &_m[0]; }

        const T* data() const { return &_m[0]; }

    private:
        T _m[D0*D1]{0};
    };

    template<typename T,
    typename = std::enable_if<std::is_floating_point_v<T>>>
    struct xquat 
    {
        xquat() = default;
        
        xquat(const xquat&) = default;

        xquat(xquat&&) = default;
        
        xquat& operator=(const xquat&) = default;
        
        xquat& operator=(xquat&&) = default;

        explicit xquat(T* array_) {
            std::memcpy(data(), array_, sizeof(T)*4);
        }

        T&       operator[](uint32_t i) { return _q[i];}

        const T& operator[](uint32_t i) const { return _q[i];}

        xquat operator*(T scalar_) const { return {_q[0]*scalar_, _q[1]*scalar_, _q[2]*scalar_, _q[3]*scalar_}; }

        xquat operator/(T scalar_) const { return {_q[0]/scalar_, _q[1]/scalar_, _q[2]/scalar_, _q[3]/scalar_}; }

        T* data() { return &_q[0]; }

        const T* data() const { return &_q[0]; }

    private:
        T _q[4]{0};
    };

    
    ////////////////////////
    // type definitions
    ///////////////////////

    template<typename T> 
    using xmtx4x4 = matrix_t<T, 4, 4>;

    using xfloat4x4  = xmtx4x4<float>;
    using xdouble4x4 = xmtx4x4<double>;

    using xquatf = xquat<float>;
    using xquatd = xquat<double>;

    template<typename T> 
    using xvec4 = vector_t<T, 4>;
    template<typename T> 
    using xvec3 = vector_t<T, 3>;
    template<typename T> 
    using xvec2 = vector_t<T, 2>;

    using xfloat4 = vector_t<float, 4>;
    using xfloat3 = vector_t<float, 3>;
    using xfloat2 = vector_t<float, 2>;

    using xdouble4 = vector_t<double, 4>;
    using xdouble3 = vector_t<double, 3>;
    using xdouble2 = vector_t<double, 2>;

    using xint4 = vector_t<int32_t, 4>;
    using xint3 = vector_t<int32_t, 3>;
    using xint2 = vector_t<int32_t, 2>;

    using xuint4 = vector_t<uint32_t, 4>;
    using xuint3 = vector_t<uint32_t, 3>;
    using xuint2 = vector_t<uint32_t, 2>;

    using xbool4 = vector_t<bool, 4>;
    using xbool3 = vector_t<bool, 3>;
    using xbool2 = vector_t<bool, 2>;

    ////////////////////////
    // binary
    ///////////////////////

    template<typename T, typename = std::enable_if_t<
    std::is_unsigned_v<T> ||
    std::is_enum_v<T> && 
    std::is_unsigned_v<std::underlying_type_t<T>>>>
    MXLIB_FUNC_DECL static bool any(T bitmask);

    template<typename T, typename = std::enable_if_t<
    std::is_unsigned_v<T> ||
    std::is_enum_v<T> && 
    std::is_unsigned_v<std::underlying_type_t<T>>>>
    MXLIB_FUNC_DECL static bool all(T bitmask);

    template<typename T, typename = std::enable_if_t<
    std::is_unsigned_v<T> ||
    std::is_enum_v<T> && 
    std::is_unsigned_v<std::underlying_type_t<T>>>>
    MXLIB_FUNC_DECL static bool contains(T mask_a, T mask_b);

    template<typename T, typename = std::enable_if_t<
    std::is_unsigned_v<T> ||
    std::is_enum_v<T> && 
    std::is_unsigned_v<std::underlying_type_t<T>>>>
    MXLIB_FUNC_DECL static T combine(T mask_a, T mask_b);
 
    ////////////////////////
    // math functions
    ///////////////////////

    template<typename T>
    MXLIB_FUNC_DECL static xmtx4x4<T> matrix_identity();
    
    template<typename T>
    MXLIB_FUNC_DECL static xmtx4x4<T> matrix_translation(const xvec3<T>& position_);

    template<typename T>
    MXLIB_FUNC_DECL static xmtx4x4<T> matrix_scale(const xvec3<T>& scale_);

    template<typename T>
    MXLIB_FUNC_DECL static xmtx4x4<T> matrix_rotation(const xquat<T>& rot_);

    template<typename T>
    MXLIB_FUNC_DECL static xmtx4x4<T> matrix_multiply(xmtx4x4<T> a, xmtx4x4<T> b);

    template<typename T>
    MXLIB_FUNC_DECL static xmtx4x4<T> compose_model_matrix(const xvec3<T>& position_, const xquat<T>& rotation_, const xvec3<T>& scale_);

    template<typename T>
    MXLIB_FUNC_DECL static xquat<T> to_quat(const xvec3<T>& euler_);
    
    template<typename T>
    MXLIB_FUNC_DECL static xvec3<T> to_euler(const xquat<T>& quat_);

    template<typename T>
    MXLIB_FUNC_DECL static xvec3<T> transform(const xvec3<T>& vec_, const xmtx4x4<T>& mtx_);

    template<typename T>
    MXLIB_FUNC_DECL static T sq_magnitude(const xquat<T>& quat_);

    template<typename T>
    MXLIB_FUNC_DECL static xquat<T> conjugate(const xquat<T>& quat_);

    template<typename T>
    MXLIB_FUNC_DECL static xquat<T> inverse(const xquat<T>& quat_);

    template<typename T>
    MXLIB_FUNC_DECL static xmtx4x4<T> inverse(const xmtx4x4<T>& mtx_);

    template<typename T>
    MXLIB_FUNC_DECL static xvec3<T> negate(const xvec3<T>& v0);

    template<typename T>
    MXLIB_FUNC_DECL static xvec3<T> multiply(const xvec3<T>& v0, float scalar);

    template<typename T>
    MXLIB_FUNC_DECL static xvec3<T> divide(const xvec3<T>& v0, float scalar);

    template<typename T>
    MXLIB_FUNC_DECL static xvec3<T> add(const xvec3<T>& v0, const xvec3<T>& v1);

    template<typename T>
    MXLIB_FUNC_DECL static xvec3<T> subtract(const xvec3<T>& v0, const xvec3<T>& v1);

    template<typename T>
    MXLIB_FUNC_DECL static T dot_product(const xvec3<T>& v0, const xvec3<T>& v1);

    template<typename T>
    MXLIB_FUNC_DECL static xvec3<T> cross_product(const xvec3<T>& v0, const xvec3<T>& v1);

    template<typename T>
    MXLIB_FUNC_DECL static T mixed_product(const xvec3<T>& v0, const xvec3<T>& v1, const xvec3<T>& v2);

    template<typename T>
    MXLIB_FUNC_DECL static xvec3<T> triple_product(const xvec3<T>& v0, const xvec3<T>& v1, const xvec3<T>& v2);
}

namespace mxlib
{
    template<typename T, uint32_t N>
    struct fixed_list
    {
        using value_type = T;

        constexpr uint32_t max_size() const {
            return N;
        }

        fixed_list() = default;

        template<typename... ARGS, 
        typename = std::enable_if_t<(sizeof...(ARGS)<=N) 
        && (std::conjunction_v<std::is_convertible<ARGS, T>...>)>>
        fixed_list(ARGS&&... args): _array(args...), _size(sizeof...(ARGS)) {}

        fixed_list(const fixed_list& other_) {
            *this = other_;
        }
        
        fixed_list(fixed_list&& other_) {
            *this = std::move(other_); // <<< to be verbose 
        }

        fixed_list& operator=(const fixed_list& other_) {
            _size = other_.size();
            std::memcpy(data(), other_.data(), sizeof(T)*_size);
            return *this;
        }

        fixed_list& operator=(fixed_list&& other_) noexcept {
            if(this != &other_) {
                _size = other_.size();
                std::memcpy(data(), other_.data(), sizeof(T)*_size);
                other_._size = 0;
            }
            return *this;
        }

        T& operator[](uint32_t i) {
            assert(i < _size);
            return _array[i];
        }

        const T& operator[](uint32_t i) const { 
            assert(i < _size); 
            return _array[i]; 
        }

        void add(const T& value_) {
            assert(_size <= N);
            _array[_size] = value_;
            ++_size;
        }
        
        void replace(uint32_t i, const T& value_) {
            assert(i < _size);
            _array[i] = value_;
        }

        void remove(uint32_t i) {
            assert(i < _size);
            for (int j = i; i < N - 1; ++i) {
                _array[i] = _array[i + 1]; 
            }
            --_size;
        }

        void     reset() { _size = 0; }

        void     swap(uint32_t i, uint32_t j) { std::swap(_array[i], _array[j]); }

        uint32_t size() const {  return _size; }

        T*       back() { return _size > 0 ? &_array[_size-1] : nullptr; }
        
        const T* back() const { return _size > 0 ? &_array[_size-1] : nullptr; }

        T*       data() { return &_array[0]; }
        
        const T* data() const { return &_array[0]; }

    private:

        T _array[N]{};
        
        uint32_t 
        _size = 0;
    };
}

template<typename T, typename>
bool mxlib::any(T bitmask)
{
    return bitmask > 0;
}

template<typename T, typename>
bool mxlib::all(T bitmask)
{
    return ~bitmask == 0;
}

template<typename T, typename>
bool mxlib::contains(T mask_a, T mask_b)
{
    return (mask_a & mask_b) == mask_b;
}

template<typename T, typename>
T mxlib::combine(T mask_a, T mask_b) 
{
    return mask_a | mask_b;
}

template<typename T>
mxlib::xmtx4x4<T> mxlib::matrix_identity()
{
    return { 
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f };
}

template<typename T>
mxlib::xmtx4x4<T> mxlib::matrix_translation(const xvec3<T>& position_)
{
    return {  
        1.0f, 0.0f, 0.0f, position_[0],
        0.0f, 1.0f, 0.0f, position_[1],
        0.0f, 0.0f, 1.0f, position_[2],
        0.0f, 0.0f, 0.0f, 1.0f };
}

template<typename T>
mxlib::xmtx4x4<T> mxlib::matrix_scale(const xvec3<T>& scale_)
{
    return {  
        scale_[0], 0.0f, 0.0f, 0.0f,
        0.0f, scale_[1], 0.0f, 0.0f,
        0.0f, 0.0f, scale_[2], 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f };
}

template<typename T>
mxlib::xmtx4x4<T> mxlib::matrix_rotation(const mxlib::xquat<T>& rot_)
{   
    // 3x3 matrix is enough, but for the purpose of this implementation this is ok
    mxlib::xmtx4x4<T> mtx_ = matrix_identity<T>();

    float a2 = rot_[0]*rot_[0];
    float b2 = rot_[1]*rot_[1];
    float c2 = rot_[2]*rot_[2];
    float ac = rot_[0]*rot_[2];
    float ab = rot_[0]*rot_[1];
    float bc = rot_[1]*rot_[2];
    float ad = rot_[3]*rot_[0];
    float bd = rot_[3]*rot_[1];
    float cd = rot_[3]*rot_[2];

    mtx_[0] = 1 - 2*(b2 + c2);
    mtx_[4] = 2*(ab + cd);
    mtx_[8] = 2*(ac - bd);

    mtx_[1] = 2*(ab - cd);
    mtx_[5] = 1 - 2*(a2 + c2);
    mtx_[9] = 2*(bc + ad);

    mtx_[2] = 2*(ac + bd);
    mtx_[6] = 2*(bc - ad);
    mtx_[10] = 1 - 2*(a2 + b2);

    return mtx_;
}

template<typename T>
mxlib::xmtx4x4<T> mxlib::matrix_multiply(mxlib::xmtx4x4<T> a, mxlib::xmtx4x4<T> b)
{
    mxlib::xmtx4x4<T> mtx_ = {};

    mtx_[0]  = a[0]*b[0]  + a[4]*b[1]  + a[8]*b[2]   + a[12]*b[3];
    mtx_[4]  = a[0]*b[4]  + a[4]*b[5]  + a[8]*b[6]   + a[12]*b[7];
    mtx_[8]  = a[0]*b[8]  + a[4]*b[9]  + a[8]*b[10]  + a[12]*b[11];
    mtx_[12] = a[0]*b[12] + a[4]*b[13] + a[8]*b[14]  + a[12]*b[15];
    mtx_[1]  = a[1]*b[0]  + a[5]*b[1]  + a[9]*b[2]   + a[13]*b[3];
    mtx_[5]  = a[1]*b[4]  + a[5]*b[5]  + a[9]*b[6]   + a[13]*b[7];
    mtx_[9]  = a[1]*b[8]  + a[5]*b[9]  + a[9]*b[10]  + a[13]*b[11];
    mtx_[13] = a[1]*b[12] + a[5]*b[13] + a[9]*b[14]  + a[13]*b[15];
    mtx_[2]  = a[2]*b[0]  + a[6]*b[1]  + a[10]*b[2]  + a[14]*b[3];
    mtx_[6]  = a[2]*b[4]  + a[6]*b[5]  + a[10]*b[6]  + a[14]*b[7];
    mtx_[10] = a[2]*b[8]  + a[6]*b[9]  + a[10]*b[10] + a[14]*b[11];
    mtx_[14] = a[2]*b[12] + a[6]*b[13] + a[10]*b[14] + a[14]*b[15];
    mtx_[3]  = a[3]*b[0]  + a[7]*b[1]  + a[11]*b[2]  + a[15]*b[3];
    mtx_[7]  = a[3]*b[4]  + a[7]*b[5]  + a[11]*b[6]  + a[15]*b[7];
    mtx_[11] = a[3]*b[8]  + a[7]*b[9]  + a[11]*b[10] + a[15]*b[11];
    mtx_[15] = a[3]*b[12] + a[7]*b[13] + a[11]*b[14] + a[15]*b[15];

    return mtx_;
}

template<typename T>
mxlib::xmtx4x4<T> mxlib::compose_model_matrix(const xvec3<T>& position_, const xquat<T>& rotation_, const xvec3<T>& scale_)
{
    xmtx4x4<T> t_mtx  = matrix_translation(position_);
    xmtx4x4<T> r_mtx  = matrix_rotation(rotation_);
    xmtx4x4<T> s_mtx  = matrix_scale(scale_); 
    
    xmtx4x4<T> trs_mtx = matrix_identity<T>();

    // Reminder: matrix multiplications are
    // not communicative operations, meaning multiplication order matters.
    // for model matrix TRS and SRT orders are both valid options to choose from.
    
    // apply rotation 
    trs_mtx = matrix_multiply(trs_mtx, r_mtx);
    // apply translation
    trs_mtx = matrix_multiply(trs_mtx, t_mtx);
    // apply scale
    trs_mtx = matrix_multiply(trs_mtx, s_mtx);
    
    return trs_mtx;
}

template<typename T>
mxlib::xquat<T> mxlib::to_quat(const xvec3<T>& euler_)
{
    mxlib::xquat<T> quat_ = {0};

    float x0 = cosf(euler_.x*0.5f);
    float x1 = sinf(euler_.x*0.5f);
    float y0 = cosf(euler_.y*0.5f);
    float y1 = sinf(euler_.y*0.5f);
    float z0 = cosf(euler_.z*0.5f);
    float z1 = sinf(euler_.z*0.5f);

    quat_[0] = x1*y0*z0 - x0*y1*z1;
    quat_[1] = x0*y1*z0 + x1*y0*z1;
    quat_[2] = x0*y0*z1 - x1*y1*z0;
    quat_[3] = x0*y0*z0 + x1*y1*z1;

    return quat_;
}

template<typename T>
mxlib::xvec3<T> mxlib::to_euler(const xquat<T>& quat_)
{
    mxlib::xvec3<T> euler_{};

    float x0 = 2.0f * (quat_[3]*quat_[0] + quat_[1]*quat_[2]);
    float x1 = 1.0f - 2.0f * (quat_[0]*quat_[0] + quat_[1]*quat_[1]);
    euler_[0] = std::atan2f(x0, x1);

    float y0 = 2.0f * (quat_[3]*quat_[1] + quat_[2]*quat_[0]);
    y0 = y0 >  1.0f ?  1.0f : y0;
    y0 = y0 < -1.0f ? -1.0f : y0;
    euler_[1] = std::asinf(y0);

    float z0 = 2.0f * (quat_[3]*quat_[2] + quat_[0]*quat_[1]);
    float z1 = 1.0f - 2.0f * (quat_[1]*quat_[1] + quat_[2]*quat_[2]);
    euler_[2] = std::atan2f(z0, z1);

    return euler_;
}

template<typename T>
mxlib::xvec3<T> mxlib::transform(const mxlib::xvec3<T>& vec_, const mxlib::xmtx4x4<T>& mtx_)
{
    T w = mtx_[12] * vec_[0] + mtx_[13] * vec_[1] + mtx_[14] * vec_[2] + mtx_[15];
    return {
        (mtx_[0] * vec_[0] + mtx_[1] * vec_[1] + mtx_[2]  * vec_[2] + mtx_[3])  / w,
        (mtx_[4] * vec_[0] + mtx_[5] * vec_[1] + mtx_[6]  * vec_[2] + mtx_[7])  / w,
        (mtx_[8] * vec_[0] + mtx_[9] * vec_[1] + mtx_[10] * vec_[2] + mtx_[11]) / w
    };
}

template<typename T>
T mxlib::sq_magnitude(const xquat<T>& quat_) 
{
    return quat_[0]*quat_[0] + quat_[1]*quat_[1] + quat_[2]*quat_[2] + quat_[3]*quat_[3];
}

template<typename T>
mxlib::xquat<T> mxlib::conjugate(const xquat<T>& quat_) 
{
    return {-quat_[0], -quat_[1], -quat_[2], quat_[3]};
}

template<typename T>
mxlib::xquat<T> mxlib::inverse(const xquat<T>& quat_) 
{
    auto sq_mag = sq_magnitude(quat_);
    assert(sq_mag != 0);
    return conjugate(quat_) / sq_mag;
}

template<typename T>
mxlib::xmtx4x4<T> mxlib::inverse(const xmtx4x4<T>& mtx_)
{
    float a00 = mtx_[0],  a01 = mtx_[1],  a02 = mtx_[2],  a03 = mtx_[3];
    float a10 = mtx_[4],  a11 = mtx_[5],  a12 = mtx_[6],  a13 = mtx_[7];
    float a20 = mtx_[8],  a21 = mtx_[9],  a22 = mtx_[10], a23 = mtx_[11];
    float a30 = mtx_[12], a31 = mtx_[13], a32 = mtx_[14], a33 = mtx_[15];

    float b00 = a00*a11 - a01*a10;
    float b01 = a00*a12 - a02*a10;
    float b02 = a00*a13 - a03*a10;
    float b03 = a01*a12 - a02*a11;
    float b04 = a01*a13 - a03*a11;
    float b05 = a02*a13 - a03*a12;
    float b06 = a20*a31 - a21*a30;
    float b07 = a20*a32 - a22*a30;
    float b08 = a20*a33 - a23*a30;
    float b09 = a21*a32 - a22*a31;
    float b10 = a21*a33 - a23*a31;
    float b11 = a22*a33 - a23*a32;

    // invert determinant
    float det = 1.0f/(b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06);

    mxlib::xmtx4x4<T> inversed = {};

    inversed[0]  = (a11*b11  - a12*b10 + a13*b09) * det;
    inversed[1]  = (-a01*b11 + a02*b10 - a03*b09) * det;
    inversed[2]  = (a31*b05  - a32*b04 + a33*b03) * det;
    inversed[3]  = (-a21*b05 + a22*b04 - a23*b03) * det;
    inversed[4]  = (-a10*b11 + a12*b08 - a13*b07) * det;
    inversed[5]  = (a00*b11  - a02*b08 + a03*b07) * det;
    inversed[6]  = (-a30*b05 + a32*b02 - a33*b01) * det;
    inversed[7]  = (a20*b05  - a22*b02 + a23*b01) * det;
    inversed[8]  = (a10*b10  - a11*b08 + a13*b06) * det;
    inversed[9]  = (-a00*b10 + a01*b08 - a03*b06) * det;
    inversed[10] = (a30*b04  - a31*b02 + a33*b00) * det;
    inversed[11] = (-a20*b04 + a21*b02 - a23*b00) * det;
    inversed[12] = (-a10*b09 + a11*b07 - a12*b06) * det;
    inversed[13] = (a00*b09  - a01*b07 + a02*b06) * det;
    inversed[14] = (-a30*b03 + a31*b01 - a32*b00) * det;
    inversed[15] = (a20*b03  - a21*b01 + a22*b00) * det;

    return inversed;
}

template<typename T>
mxlib::xvec3<T> mxlib::negate(const xvec3<T>& v0) {
    return {-v0[0], -v0[1], -v0[2]}; 
}

template<typename T>
mxlib::xvec3<T> mxlib::multiply(const xvec3<T>& v0, float scalar) {
    return {
        v0[0]*scalar,
        v0[1]*scalar,
        v0[2]*scalar,
    };
}

template<typename T>
mxlib::xvec3<T> mxlib::divide(const xvec3<T>& v0, float scalar) {
    return {
        v0[0]/scalar,
        v0[1]/scalar,
        v0[2]/scalar,
    };
}

template<typename T>
mxlib::xvec3<T> mxlib::add(const xvec3<T>& v0, const xvec3<T>& v1) {
    return {
        v0[0]+v1[0],
        v0[1]+v1[1],
        v0[2]+v1[2],
    };
}

template<typename T>
mxlib::xvec3<T> mxlib::subtract(const xvec3<T>& v0, const xvec3<T>& v1) {
    return {
        v0[0]-v1[0],
        v0[1]-v1[1],
        v0[2]-v1[2],
    };
}

template<typename T>
T mxlib::dot_product(const xvec3<T>& v0, const xvec3<T>& v1) {
    return v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2]; 
}

template<typename T>
mxlib::xvec3<T> mxlib::cross_product(const xvec3<T>& v0, const xvec3<T>& v1) {
    return {
        v0[1]*v1[2] - v0[2]*v1[1],
        v0[2]*v1[0] - v0[0]*v1[2],
        v0[0]*v1[1] - v0[1]*v1[0],
    };
}

template<typename T>
T mxlib::mixed_product(const xvec3<T>& v0, const xvec3<T>& v1, const xvec3<T>& v2) {
    return dot_product<T>(cross_product<T>(v0, v1), v2);
}

template<typename T>
mxlib::xvec3<T> mxlib::triple_product(const xvec3<T>& v0, const xvec3<T>& v1, const xvec3<T>& v2) {
    return cross_product<T>(cross_product<T>(v0, v1), v2);
}