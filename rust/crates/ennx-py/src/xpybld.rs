#[path = "link_rpath.rs"]
mod link_rpath;

include!("xpybld_api.inc.rs");
ennx_api!(link_rpath);
