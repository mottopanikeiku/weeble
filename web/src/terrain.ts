export const TERRAIN_Y = 12;

/** Translation only: keep the training height samples, dimensions and friction. */
export function addTerrain(xml: string, training: string, alreadyRough: boolean): string {
  const hfield = /<hfield\b[^>]*\/>/.exec(training)?.[0];
  const floor = /<geom\b[^>]*\bname="floor"[^>]*\/>/.exec(training)?.[0];
  if (!hfield || !floor) throw new Error('Missing exported rough ground');
  const place = (tag: string) => tag.replace(/\spos="[^"]*"/, '').replace('/>', ` pos="0 ${TERRAIN_Y} 0"/>`);
  let result = xml;
  let asset = '';
  let surface = '';
  let pairs = '';
  if (alreadyRough) {
    result = result.replace(/<geom\b[^>]*\bname="floor"[^>]*\/>/, place);
    // Render the surrounding ground, but give physics a hole under the hfield.
    // This avoids a second contact surface at the field's lowest samples.
    surface = `<geom name="play_backdrop" type="plane" size="0 0 .01" contype="0" conaffinity="0" rgba=".6 .65 .5 1"/>
      <geom name="play_outer_south" type="box" pos="0 -49 -.05" size="100 51 .05" contype="1" conaffinity="0" friction=".6" group="3"/>
      <geom name="play_outer_north" type="box" pos="0 71 -.05" size="100 49 .05" contype="1" conaffinity="0" friction=".6" group="3"/>
      <geom name="play_outer_west" type="box" pos="-55 12 -.05" size="45 10 .05" contype="1" conaffinity="0" friction=".6" group="3"/>
      <geom name="play_outer_east" type="box" pos="55 12 -.05" size="45 10 .05" contype="1" conaffinity="0" friction=".6" group="3"/>`;
  } else {
    asset = hfield.replace('name="hfield"', 'name="play_hfield"')
      .replace(/file="[^"]+"/, 'file="/assets/go1/assets/hfield.png"');
    surface = place(floor.replace('name="floor"', 'name="play_terrain"')
      .replace('hfield="hfield"', 'hfield="play_hfield"').replace(/\smaterial="[^"]*"/, ''));
    // G1 uses explicit foot-floor pairs, not the Go1 bitmask convention.
    pairs = Array.from(xml.matchAll(/<pair\b[^>]*\/>/g), match => match[0])
      .filter(tag => /geom[12]="floor"/.test(tag))
      .map(tag => tag.replace(/\sname="[^"]*"/, '').replace(/(geom[12]=)"floor"/, '$1"play_terrain"')).join('\n');
  }
  surface += `<geom name="play_prop_terrain" type="hfield" hfield="${alreadyRough ? 'hfield' : 'play_hfield'}" pos="0 ${TERRAIN_Y} 0" contype="8" conaffinity="4" friction="1" priority="1" group="3"/>`;
  return result.replace('</mujoco>', `<asset>${asset}</asset><worldbody>${surface}</worldbody><contact>${pairs}</contact></mujoco>`);
}
