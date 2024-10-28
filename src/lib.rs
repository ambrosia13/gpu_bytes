trait AsGpuBytes {
    fn as_gpu_bytes(&self) -> GpuBytes;
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    #[default]
    Std140,
    Std430,
}

#[derive(Debug, Default, Clone)]
struct GpuBytes {
    bytes: Vec<u8>,
    alignment: usize,
    layout: Layout,
}

impl GpuBytes {
    pub fn new(layout: Layout) -> Self {
        Self {
            layout,
            ..Default::default()
        }
    }

    fn write_slice(&mut self, data: &[u8], align: usize) {
        self.alignment = self.alignment.max(align);

        let offset = self.bytes.len();
        let padding = (align - (offset % align)) % align;

        self.bytes.extend(std::iter::repeat(0u8).take(padding));

        self.bytes.extend_from_slice(data);
    }

    pub fn write<T: AsGpuBytes>(&mut self, data: &T) -> &mut Self {
        let data = data.as_gpu_bytes();

        self.write_slice(data.as_slice(), data.alignment);
        self
    }

    pub fn write_array<T: AsGpuBytes>(&mut self, data: impl IntoIterator<Item = T>) -> &mut Self {
        for elem in data.into_iter() {
            let mut elem = elem.as_gpu_bytes();

            match self.layout {
                Layout::Std140 => {
                    elem.align_to(elem.alignment.next_multiple_of(16));
                }
                Layout::Std430 => {
                    elem.align();
                }
            }

            self.write(&elem);
        }

        self
    }

    pub fn align(&mut self) -> &mut Self {
        self.align_to(self.alignment)
    }

    pub fn align_to(&mut self, align: usize) -> &mut Self {
        let offset = self.bytes.len();
        let padding = (align - (offset % align)) % align;

        self.bytes.extend(std::iter::repeat(0u8).take(padding));
        self.alignment = align;
        self
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

impl AsGpuBytes for GpuBytes {
    fn as_gpu_bytes(&self) -> GpuBytes {
        self.clone()
    }
}

pub trait AsStd140 {
    fn as_std140(&self) -> Std140Bytes;
}

#[derive(Debug, Clone)]
pub struct Std140Bytes {
    gpu_bytes: GpuBytes,
}

impl Std140Bytes {
    pub fn new() -> Self {
        Self {
            gpu_bytes: GpuBytes::new(Layout::Std140),
        }
    }

    pub fn write<T: AsStd140>(&mut self, data: &T) -> &mut Self {
        self.gpu_bytes.write(&data.as_std140().gpu_bytes);
        self
    }

    pub fn write_array<T: AsStd140>(&mut self, data: &[T]) -> &mut Self {
        self.gpu_bytes
            .write_array(data.iter().map(|e| e.as_std140().gpu_bytes));
        self
    }

    pub fn align(&mut self) -> &mut Self {
        self.gpu_bytes.align();
        self
    }

    pub fn align_to(&mut self, align: usize) -> &mut Self {
        let offset = self.gpu_bytes.bytes.len();
        let padding = (align - (offset % align)) % align;

        self.gpu_bytes
            .bytes
            .extend(std::iter::repeat(0u8).take(padding));
        self.gpu_bytes.alignment = align;
        self
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.gpu_bytes.bytes
    }
}

impl Default for Std140Bytes {
    fn default() -> Self {
        Self::new()
    }
}

impl AsStd140 for Std140Bytes {
    fn as_std140(&self) -> Std140Bytes {
        self.clone()
    }
}

pub trait AsStd430 {
    fn as_std430(&self) -> Std430Bytes;
}

#[derive(Debug, Clone)]
pub struct Std430Bytes {
    gpu_bytes: GpuBytes,
}

impl Std430Bytes {
    pub fn new() -> Self {
        Self {
            gpu_bytes: GpuBytes::new(Layout::Std430),
        }
    }

    pub fn write<T: AsStd430>(&mut self, data: &T) -> &mut Self {
        self.gpu_bytes.write(&data.as_std430().gpu_bytes);
        self
    }

    pub fn write_array<T: AsStd430>(&mut self, data: &[T]) -> &mut Self {
        self.gpu_bytes
            .write_array(data.iter().map(|e| e.as_std430().gpu_bytes));
        self
    }

    pub fn align(&mut self) -> &mut Self {
        self.gpu_bytes.align();
        self
    }

    pub fn align_to(&mut self, align: usize) -> &mut Self {
        let offset = self.gpu_bytes.bytes.len();
        let padding = (align - (offset % align)) % align;

        self.gpu_bytes
            .bytes
            .extend(std::iter::repeat(0u8).take(padding));
        self.gpu_bytes.alignment = align;
        self
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.gpu_bytes.bytes
    }
}

impl Default for Std430Bytes {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! primitive_impl_std140_std430 {
    ($datatype:ty, align = $align:literal) => {
        impl AsStd140 for $datatype {
            fn as_std140(&self) -> Std140Bytes {
                let mut buf = Std140Bytes::new();

                const SIZE: usize = std::mem::size_of::<$datatype>();
                let cast: [u8; SIZE] = bytemuck::cast(*self);

                buf.gpu_bytes.bytes.extend_from_slice(&cast);
                buf.gpu_bytes.alignment = $align;

                buf
            }
        }

        impl AsStd430 for $datatype {
            fn as_std430(&self) -> Std430Bytes {
                let mut buf = Std430Bytes::new();

                const SIZE: usize = std::mem::size_of::<$datatype>();
                let cast: [u8; SIZE] = bytemuck::cast(*self);

                buf.gpu_bytes.bytes.extend_from_slice(&cast);
                buf.gpu_bytes.alignment = $align;

                buf
            }
        }
    };
}

macro_rules! primitive_impl_std140_std430_matrix {
    ($datatype:ty, columns = $columns:literal) => {
        impl AsStd140 for $datatype {
            fn as_std140(&self) -> Std140Bytes {
                let mut buf = Std140Bytes::new();

                for i in 0..$columns {
                    buf.write(&self.col(i));
                }

                buf
            }
        }

        impl AsStd430 for $datatype {
            fn as_std430(&self) -> Std430Bytes {
                let mut buf = Std430Bytes::new();

                for i in 0..$columns {
                    buf.write(&self.col(i));
                }

                buf
            }
        }
    };
}

primitive_impl_std140_std430!(f32, align = 4);
primitive_impl_std140_std430!(glam::Vec2, align = 8);
primitive_impl_std140_std430!(glam::Vec3, align = 16);
primitive_impl_std140_std430!(glam::Vec4, align = 16);

primitive_impl_std140_std430!(i32, align = 4);
primitive_impl_std140_std430!(glam::IVec2, align = 8);
primitive_impl_std140_std430!(glam::IVec3, align = 16);
primitive_impl_std140_std430!(glam::IVec4, align = 16);

primitive_impl_std140_std430!(u32, align = 4);
primitive_impl_std140_std430!(glam::UVec2, align = 8);
primitive_impl_std140_std430!(glam::UVec3, align = 16);
primitive_impl_std140_std430!(glam::UVec4, align = 16);

primitive_impl_std140_std430_matrix!(glam::Mat3, columns = 3);
primitive_impl_std140_std430_matrix!(glam::Mat4, columns = 4);

impl<T: AsStd140 + Default> AsStd140 for Vec<T> {
    fn as_std140(&self) -> Std140Bytes {
        let mut buf = Std140Bytes::new();

        if self.capacity() == 0 {
            panic!("A Vec<T> should have an initial capacity before being converted to gpu layout");
        }

        let mut std140 = T::default().as_std140();

        // in std140, array elements are aligned to a multiple of 16
        std140.align_to(16);

        let bytes_per_element = std140.as_slice().len();

        // the gpu representation will contain as many bytes as possible to hold the vec's capacity
        // and fill the appropriate number of bytes with the vec's elements
        let total_bytes = bytes_per_element * self.capacity();

        for elem in self.iter() {
            let mut std140 = elem.as_std140();

            // in std140, array elements are aligned to a multiple of 16
            std140.align_to(16);

            buf.gpu_bytes.bytes.extend_from_slice(std140.as_slice());
        }

        // now pad with 0's for the remaining capacity
        let padding = total_bytes - buf.gpu_bytes.bytes.len();

        buf.gpu_bytes
            .bytes
            .extend(std::iter::repeat(0u8).take(padding));

        buf.gpu_bytes.alignment = 16;

        buf
    }
}

impl<T: AsStd430 + Default> AsStd430 for Vec<T> {
    fn as_std430(&self) -> Std430Bytes {
        let mut buf = Std430Bytes::new();

        if self.capacity() == 0 {
            panic!("A Vec<T> should have an initial capacity before being converted to gpu layout");
        }

        let mut std430 = T::default().as_std430();
        std430.align();

        let bytes_per_element = std430.as_slice().len();

        // the gpu representation will contain as many bytes as possible to hold the vec's capacity
        // and fill the appropriate number of bytes with the vec's elements
        let total_bytes = bytes_per_element * self.capacity();

        for elem in self.iter() {
            let mut std430 = elem.as_std430();
            std430.align();

            buf.gpu_bytes.bytes.extend_from_slice(std430.as_slice());
        }

        // now pad with 0's for the remaining capacity
        let padding = total_bytes - buf.gpu_bytes.bytes.len();

        buf.gpu_bytes
            .bytes
            .extend(std::iter::repeat(0u8).take(padding));

        // the alignment of the array is the same as the alignment of the elements in std430
        buf.gpu_bytes.alignment = std430.gpu_bytes.alignment;

        buf
    }
}

#[cfg(test)]
mod tests {
    use glam::UVec3;

    use super::*;

    #[test]
    fn std140_vec3_and_scalar() {
        let mut buf = Std140Bytes::new();

        buf.write(&UVec3::splat(u32::MAX));
        buf.write(&u32::MAX);
        buf.align();

        #[rustfmt::skip]
        assert_eq!(
            buf.as_slice(),
            &[
                // x
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // y
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // z
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // scalar
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
            ]
        );
    }

    #[test]
    fn std140_vec3_and_vec3() {
        let mut buf = Std140Bytes::new();

        buf.write(&UVec3::splat(u32::MAX));
        buf.write(&UVec3::splat(u32::MAX));
        buf.align();

        #[rustfmt::skip]
        assert_eq!(
            buf.as_slice(),
            &[
                // x
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // y
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // z
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // padding
                0, 0, 0, 0, 
                // x
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // y
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // z
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // padding
                0, 0, 0, 0, 
            ]
        );
    }

    #[test]
    fn std140_scalar_and_vec3() {
        let mut buf = Std140Bytes::new();

        buf.write(&u32::MAX);
        buf.write(&UVec3::splat(u32::MAX));
        buf.align();

        #[rustfmt::skip]
        assert_eq!(
            buf.as_slice(),
            &[
                // scalar
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // padding
                0, 0, 0, 0, 
                // padding
                0, 0, 0, 0, 
                // padding
                0, 0, 0, 0, 
                // x
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // y
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // z
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // padding
                0, 0, 0, 0, 
            ]
        );
    }

    #[test]
    fn std140_scalar_array() {
        let mut buf = Std140Bytes::new();

        buf.write_array(&[u32::MAX, u32::MAX]);
        buf.align();

        #[rustfmt::skip]
        assert_eq!(
            buf.as_slice(),
            &[
                // scalar
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // padding
                0, 0, 0, 0, 
                // padding
                0, 0, 0, 0, 
                // padding
                0, 0, 0, 0, 
                // scalar
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // padding
                0, 0, 0, 0, 
                // padding
                0, 0, 0, 0, 
                // padding
                0, 0, 0, 0, 
            ]
        );
    }

    #[test]
    fn std430_scalar_array() {
        let mut buf = Std430Bytes::new();

        buf.write_array(&[u32::MAX, u32::MAX]);
        buf.align();

        #[rustfmt::skip]
        assert_eq!(
            buf.as_slice(),
            &[
                // scalar
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
                // scalar
                u8::MAX, u8::MAX, u8::MAX, u8::MAX,
            ]
        );
    }
}
