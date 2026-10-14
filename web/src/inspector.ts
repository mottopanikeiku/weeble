import type { Controller } from './policy';

/** A read-only view: never calls observe/forward or changes policy history. */
export class Inspector {
  private controller: Controller | null = null;
  private observations: { bars: HTMLElement[]; offset: number }[] = [];
  private actions: HTMLElement[] = [];
  private contacts: { element: HTMLElement; geom: number; touching: boolean }[] = [];
  constructor(private readonly element: HTMLElement) {}

  update(controller: Controller, fallen: boolean): void {
    if (this.element.hidden) return;
    if (this.controller !== controller) this.bind(controller);
    this.element.querySelector<HTMLElement>('.inspector-state')!.textContent = fallen
      ? 'No active policy · motors hold the handoff pose' : 'Actor inputs, normalized · bars clip at ±3σ';
    const spec = controller.policy.spec, key = spec.policy.obs_key;
    for (const row of this.observations) {
      for (let i = 0; i < row.bars.length; i++) {
        const index = row.offset + i, raw = controller.observation[index];
        const normalized = spec.normalizer.enabled
          ? (raw - spec.normalizer.mean[key][index]) / spec.normalizer.std[key][index] : raw;
        const bar = row.bars[i];
        bar.style.setProperty('--value', String(Math.min(1, Math.abs(normalized) / 3)));
        bar.classList.toggle('negative', normalized < 0);
        bar.title = `Input ${index}: ${raw.toFixed(3)} (${normalized.toFixed(2)} normalized)`;
      }
    }
    for (let i = 0; i < this.actions.length; i++) {
      const value = fallen ? 0 : controller.lastAction[i];
      this.actions[i].style.setProperty('--value', String(Math.abs(value)));
      this.actions[i].classList.toggle('negative', value < 0);
      this.actions[i].title = `${this.actions[i].dataset.name}: ${fallen ? 'no policy action' : value.toFixed(3)}`;
    }
    for (const foot of this.contacts) foot.touching = false;
    const data = controller.physics.data, contacts = data.contact;
    for (let i = 0; i < data.ncon; i++) {
      const contact = contacts.get(i)!;
      for (const foot of this.contacts) {
        if (contact.geom1 === foot.geom || contact.geom2 === foot.geom) foot.touching = true;
      }
      contact.delete();
    }
    contacts.delete();
    for (const contact of this.contacts) {
      const touching = contact.touching;
      contact.element.classList.toggle('touching', touching);
      contact.element.textContent = `${contact.element.dataset.name} ${touching ? 'on' : 'off'}`;
    }
  }

  private bind(controller: Controller): void {
    this.controller = controller;
    this.element.replaceChildren();
    const heading = document.createElement('h2');
    heading.textContent = 'Inside the policy';
    const state = document.createElement('p');
    state.className = 'inspector-state';
    this.element.append(heading, state);
    const row = (name: string, count: number): HTMLElement[] => {
      const line = document.createElement('div');
      line.className = 'signal-row';
      const label = document.createElement('span');
      label.textContent = name.replaceAll('_', ' ');
      const bars = document.createElement('div');
      bars.className = 'signal-bars';
      const items = Array.from({ length: count }, () => {
        const bar = document.createElement('i');
        bars.append(bar);
        return bar;
      });
      line.append(label, bars);
      this.element.append(line);
      return items;
    };
    this.observations = controller.policy.spec.observation.fields.map(field => ({
      bars: row(field.name, field.size), offset: field.offset,
    }));
    const actionHeading = document.createElement('p');
    actionHeading.textContent = 'Actions · one bar per motor, −1 to +1';
    this.element.append(actionHeading);
    this.actions = row('position targets', controller.lastAction.length);
    const { mj, model } = controller.physics;
    this.actions.forEach((bar, i) => {
      bar.dataset.name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR.value, i);
      bar.setAttribute('aria-label', bar.dataset.name);
    });
    const legend = document.createElement('p');
    legend.textContent = 'Green + / ochre − · foot contacts below';
    const contacts = document.createElement('div');
    contacts.className = 'foot-contacts';
    this.contacts = controller.policy.spec.feet_geoms.map(name => {
      const element = document.createElement('span');
      element.dataset.name = name.replace('_foot', '');
      const geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, name);
      contacts.append(element);
      return { element, geom, touching: false };
    });
    this.element.append(legend, contacts);
  }
}
