macro_rules! define_ennx_py_build_api {
    ($link:ident) => {
        pub fn run_ennx_py_build() {
            $link::emit_linux_rpath_link_args();
        }

        pub fn main() {
            run_ennx_py_build();
        }
    };
}
