// The dream between the player's words: time goes on. Every quarter of a minute or so one
// small thing happens: a being goes to another, circles it, follows it, plays or rests; an
// object glows, grows, shrinks, stretches, melts or spins. The storyteller's director writes
// these for the story (see services/director /direct); when it has nothing to say, the
// dream invents a gentle one from what is there. The camera goes to look each time.

// Beings move about; scenery stays where it grew and only changes shape or light.
const BEINGS = new Set(['fairy', 'butterfly', 'bird', 'fish', 'cat', 'dog', 'wolf', 'horse', 'dragon', 'person',
  'ghost', 'fireflies', 'spaceship', 'boat', 'orb', 'lantern']);
const SCENERY = new Set(['tree', 'flower', 'mushroom', 'rock', 'water', 'house', 'castle', 'fire', 'rainbow',
  'cloud', 'sun', 'moon', 'planet', 'star', 'volcano', 'portal']);
const MOVES = ['approach', 'approach', 'circle', 'play', 'follow', 'rest', 'wander', 'look'];
const CHANGES = ['glow', 'spin', 'grow', 'shrink', 'stretch', 'melt', 'rise'];
export const MOVING = new Set(['approach', 'circle', 'follow', 'flee', 'wander', 'rest', 'play', 'look', 'rise',
  'sink', 'return']);

const pick = (a) => a[Math.floor(Math.random() * a.length)];

export class Life {
  constructor(world) {
    this.world = world;
    this.queue = [];
    this.next = 10;          // the first beat waits for the dream to settle
    this.asked = -1e9;
    this.count = 0;
    this.last = '';
    this.onNeed = null;      // the page asks the director for the next minute
  }

  // The director's next minute (it replaces whatever was left of the last one).
  plan(beats) {
    this.queue = (beats || []).filter((b) => b && b.who && b.does);
    this.asked = -1e9;
  }

  // The player spoke: their words come first, so the camera goes back to the dreamer and
  // the old plan is dropped (the director writes a new one after the answer).
  turn(t) {
    this.queue = [];
    this.next = t + 14;
    this.world.dream.comeBack();
  }

  kindOf(e) { return e.spec.kind === 'other' ? (e.shape || 'other') : e.spec.kind; }

  find(id) { return id === 'self' ? this.world.self : this.world.entities.get(id); }

  // A gentle beat from the scene itself.
  invent() {
    const all = [...this.world.entities.values()].filter((e) => !e.dying && e.b.members.length);
    if (!all.length) return null;
    const beings = all.filter((e) => BEINGS.has(this.kindOf(e)) && (e.spec.count || 1) <= 6);
    const things = all.filter((e) => !BEINGS.has(this.kindOf(e)) && !SCENERY.has(this.kindOf(e)));
    const scenery = all.filter((e) => SCENERY.has(this.kindOf(e)));
    const fresh = (list) => list.filter((e) => e.id !== this.last).concat(list).slice(0, Math.max(1, list.length));
    const r = Math.random();
    if (beings.length && r < 0.6) {
      const who = pick(fresh(beings));
      const others = [...beings, ...things].filter((e) => e !== who);
      const to = Math.random() < 0.35 || !others.length ? 'self' : pick(others).id;
      return { who: who.id, does: pick(MOVES), to };
    }
    if (things.length && r < 0.9) return { who: pick(fresh(things)).id, does: pick(CHANGES), to: '' };
    if (scenery.length) return { who: pick(scenery).id, does: pick(['glow', 'grow', 'stretch', 'glow']), to: '' };
    if (beings.length) return { who: pick(beings).id, does: pick(MOVES), to: 'self' };
    return null;
  }

  update(t) {
    const w = this.world;
    // Nothing new starts while the dreamer listens, thinks or speaks.
    if (w.thinking || w.voice > 0.06) { this.next = Math.max(this.next, t + 5); return; }
    if (t < this.next) return;
    this.next = t + 15 + Math.random() * 9;
    const beat = this.queue.shift() || this.invent();
    if (!this.queue.length && this.onNeed && t - this.asked > 30) { this.asked = t; this.onNeed(); }
    if (beat) this.play(beat, t);
  }

  play(b, t) {
    const who = this.find(b.who);
    if (!who) return;
    const to = b.to ? this.find(b.to) : null;
    // Scenery does not walk: a move asked of it becomes a change of light.
    const does = SCENERY.has(this.kindOf(who)) && MOVING.has(b.does) ? 'glow' : b.does;
    who.act(does, to === who ? null : to, t, 24);
    this.last = who.id;
    // The camera goes to look: at both when one goes to another, now and then from afar.
    this.count++;
    const points = [who.position];
    if (to && to !== who) points.push(to.position);
    this.world.dream.frame(points, { wide: this.count % 4 === 0 });
  }
}
