macro_rules! ennx_api {
    ($link:ident) => {
        pub fn ennx_build() {
            $link::emit_args();
        }

        pub fn main() {
            ennx_build();
        }
    };
}
