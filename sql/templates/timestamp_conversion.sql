alter table sensors.gesamtmessung
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.vk_2_eg
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.stahl_folder
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.og_3
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.eg
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.entsorgung
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.og
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.uv_eg
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.r707lv_f4032
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.uv_sigma_line_eg
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.r707lv_trockner
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.r707lv_vari_air
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.xl106_druckmaschine
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.xl106_uv_scan
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.hauptluftung
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.vk_1_ug
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.r707lv_f4034
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.og_2
alter column t type timestamp using to_timestamp(t::double precision);

alter table sensors.not_in_list
alter column t type timestamp using to_timestamp(t::double precision);
