# `pair`

Header File: `Kokkos_Pair.hpp`

An implementation of `std::pair` that is intended to be fully compatible, with the exception that `Kokkos::pair` will work on the device. Also provides utility functions to convert from and to `std::pair`

Usage: 
```c++
std::pair<int, float> std_pair = std::make_pair(1,2.0f); 
Kokkos::pair<int_float> kokkos_pair = Kokkos::make_pair(1,2.0f);
Kokkos::pair<int, float> converted_std_pair(std_pair);
std::pair<int,float> converted_kokkos_pair = kokkos_pair.to_std_pair();
```

## Synopsis 
```c++
template <class T1, class T2>
struct pair {

    typedef T1 first_type;
    typedef T2 second_type;

    first_type first;
    second_type second;
  
    KOKKOS_DEFAULTED_FUNCTION constexpr pair() = default;
    KOKKOS_FORCEINLINE_FUNCTION constexpr pair(first_type const& f,
                                               second_type const& s);
  
    template <class U, class V>
    KOKKOS_FORCEINLINE_FUNCTION constexpr pair(const pair<U, V>& p);

    template <class U, class V>
    KOKKOS_FORCEINLINE_FUNCTION pair<T1, T2>& operator=(const pair<U, V>& p);
  
    template <class U, class V>
    pair(const std::pair<U, V>& p);
  
    std::pair<T1, T2> to_std_pair() const;
};
```

### Public Class Members

  * `first`: the first element in the pair
  * `second`: the second element in the pair

### Typedefs
   
  * `first_type`: the type of the first element in the pair
  * `second_type`: the type of the second element in the pair

### Constructors

  * 
  ```c++ 
  KOKKOS_DEFAULTED_FUNCTION constexpr pair() = default;
  ```

  Default constructor. Initializes both data members with their defaults

  * 
  ```c++
  KOKKOS_FORCEINLINE_FUNCTION constexpr pair(first_type const& f,
                                  second_type const& s);
  ```

  Element-wise constructor. Assigns `first` the value of `f`, `second` the value of `s` 

  * 
  ```c++
  template <class U, class V>
  KOKKOS_FORCEINLINE_FUNCTION constexpr pair(const pair<U, V>& p);
  ``` 
      
  Conversion from `std::pair`. Assigns each element of the pair to its corresponding element in the `p`

### Assignment and conversion

  * 
  ```c++
  template <class U, class V>
  KOKKOS_FORCEINLINE_FUNCTION pair<T1, T2>& operator=(const pair<U, V>& p);
  ```

  Sets `first` to `p.first` and `second` to `p.second` 
  ### Functions

  * 
  ```c++
  std::pair<T1, T2> to_std_pair() const;
  ```

  Returns a `std::pair` whose contents match those of the `Kokkos::pair`. Useful for interacting with libraries that explicitly only accept `std::pair`
