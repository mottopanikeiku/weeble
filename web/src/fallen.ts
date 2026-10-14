/** Keep the original collision hull inputs; only visible meshes are simplified. */
export function fallenAssets(xml: string): { xml: string; files: string[] } {
  const definitions: Record<string, { tag: string; file: string }> = {};
  for (const match of xml.matchAll(/<mesh\b[^>]*\/>/g)) {
    const name = /\bname="([^"]+)"/.exec(match[0])?.[1];
    const file = /\bfile="([^"]+)"/.exec(match[0])?.[1];
    if (name && file) definitions[name] = { tag: match[0], file };
  }
  const used = new Set<string>();
  const transformed = xml.replace(/<geom\b[^>]*\/>/g, tag => {
    if (!/\bname="fallen_/.test(tag)) return tag;
    const mesh = /\bmesh="([^"]+)"/.exec(tag)?.[1];
    if (!mesh) return tag;
    if (!definitions[mesh]) throw new Error(`Missing fallen collision mesh ${mesh}`);
    used.add(mesh);
    return tag.replace(`mesh="${mesh}"`, `mesh="full_${mesh}"`);
  });
  const assets = Array.from(used, name => definitions[name].tag
    .replace(`name="${name}"`, `name="full_${name}"`)
    .replace(`file="${definitions[name].file}"`, `file="full/${definitions[name].file}"`)).join('\n');
  return {
    xml: transformed.replace('</asset>', `${assets}</asset>`),
    files: Array.from(used, name => `assets/g1/assets/full/${definitions[name].file}`),
  };
}
