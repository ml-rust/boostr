//! Random schemas, valid instances and invalid mutations, for tests only.
//!
//! Schemas come from the supported subset with no any-value position, so
//! each compiles fast. An instance is a tree whose `Atom` leaves print in
//! compact form, because `enum` and `const` members match only that form.

use serde_json::{Map, Value, json};

use crate::inference::grammar::reference::Lcg;

/// A generated instance.
#[derive(Clone, Debug)]
pub(super) enum Inst {
    /// A value printed compact: a scalar or an `enum`/`const` member.
    Atom(Value),
    /// An object, members in emission order.
    Obj(Vec<(String, Inst)>),
    /// An array.
    Arr(Vec<Inst>),
}

/// The `enum` and `const` member pool.
fn pool() -> Vec<Value> {
    vec![
        json!("red"),
        json!("a\"b"),
        json!(1),
        json!(-2.5),
        json!(true),
        json!(null),
        json!({"k": [1, 2]}),
        json!([1, "x"]),
    ]
}

/// A random schema nesting at most `depth` levels.
pub(super) fn schema(rng: &mut Lcg, depth: u32) -> Value {
    let kind = if depth == 0 || rng.below_u64(3) == 0 {
        rng.below_u64(7)
    } else {
        7 + rng.below_u64(4)
    };
    let mut out = match kind {
        0 => json!({"type": "null"}),
        1 => json!({"type": "boolean"}),
        2 => json!({"type": "integer"}),
        3 => json!({"type": "number"}),
        4 => {
            let lo = rng.below_u64(3);
            let hi = lo + rng.below_u64(4);
            match rng.below_u64(3) {
                0 => json!({"type": "string"}),
                1 => json!({"type": "string", "minLength": lo}),
                _ => json!({"type": "string", "minLength": lo, "maxLength": hi}),
            }
        }
        5 => {
            let mut members = pool();
            let keep = 1 + rng.below_u64(4) as usize;
            let start = rng.below_u64(members.len() as u64) as usize;
            members.rotate_left(start);
            members.truncate(keep);
            json!({"enum": members})
        }
        6 => json!({"const": rng.pick(&pool())}),
        7 => object(rng, depth),
        8 => {
            let mut map = Map::new();
            map.insert("type".into(), json!("array"));
            map.insert("items".into(), schema(rng, depth - 1));
            if rng.below_u64(2) == 0 {
                let lo = rng.below_u64(3);
                map.insert("minItems".into(), json!(lo));
                map.insert("maxItems".into(), json!(lo + rng.below_u64(3)));
            }
            Value::Object(map)
        }
        9 => json!({"anyOf": [schema(rng, depth - 1), schema(rng, depth - 1)]}),
        _ => json!({"type": rng.pick(&[json!(["string", "null"]), json!(["integer", "boolean"])])}),
    };
    if rng.below_u64(6) == 0
        && let Some(map) = out.as_object_mut()
    {
        map.insert("nullable".into(), json!(true));
    }
    out
}

/// An object schema with 1 to 4 properties `p0`, `p1`, ... and no extra keys.
fn object(rng: &mut Lcg, depth: u32) -> Value {
    let mut props = Map::new();
    let mut required = Vec::new();
    for i in 0..1 + rng.below_u64(4) {
        let key = format!("p{i}");
        props.insert(key.clone(), schema(rng, depth - 1));
        if rng.below_u64(2) == 0 {
            required.push(json!(key));
        }
    }
    json!({"type": "object", "properties": props, "required": required, "additionalProperties": false})
}

/// A random instance valid for `schema`.
pub(super) fn instance(rng: &mut Lcg, schema: &Value) -> Inst {
    let map = schema.as_object().cloned().unwrap_or_default();
    if map.get("nullable") == Some(&json!(true)) && rng.below_u64(4) == 0 {
        return Inst::Atom(Value::Null);
    }
    if let Some(Value::Array(members)) = map.get("enum") {
        return Inst::Atom(rng.pick(members));
    }
    if let Some(value) = map.get("const") {
        return Inst::Atom(value.clone());
    }
    if let Some(Value::Array(members)) = map.get("anyOf") {
        let member = rng.pick(members);
        return instance(rng, &member);
    }
    let name = match map.get("type") {
        Some(Value::Array(names)) => rng.pick(names),
        Some(name) => name.clone(),
        None => json!("null"),
    };
    match name.as_str().unwrap_or("null") {
        "boolean" => Inst::Atom(json!(rng.below_u64(2) == 0)),
        "integer" => Inst::Atom(json!(rng.below_u64(2001) as i64 - 1000)),
        "number" => Inst::Atom(rng.pick(&[
            json!(0),
            json!(-0.0),
            json!(1.5),
            json!(-2.5e-5),
            json!(1e21),
            json!(42),
        ])),
        "string" => Inst::Atom(Value::String(text(rng, &map))),
        "object" => {
            let props = map
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let required = map.get("required").cloned().unwrap_or(json!([]));
            let mut members = Vec::new();
            for (key, sub) in &props {
                if required.as_array().is_some_and(|r| r.contains(&json!(key)))
                    || rng.below_u64(2) == 0
                {
                    members.push((key.clone(), instance(rng, sub)));
                }
            }
            Inst::Obj(members)
        }
        "array" => {
            let lo = map.get("minItems").and_then(Value::as_u64).unwrap_or(0);
            let hi = map
                .get("maxItems")
                .and_then(Value::as_u64)
                .unwrap_or(lo + 3);
            let len = lo + rng.below_u64(hi - lo + 1);
            let items = map.get("items").cloned().unwrap_or(json!({}));
            Inst::Arr((0..len).map(|_| instance(rng, &items)).collect())
        }
        _ => Inst::Atom(Value::Null),
    }
}

/// A random string that meets the length bounds of `map`.
fn text(rng: &mut Lcg, map: &Map<String, Value>) -> String {
    let chars = [
        'a',
        'Z',
        ' ',
        '"',
        '\\',
        '\n',
        '\u{1}',
        '/',
        'é',
        '世',
        '\u{10000}',
    ];
    let lo = map.get("minLength").and_then(Value::as_u64).unwrap_or(0);
    let hi = map
        .get("maxLength")
        .and_then(Value::as_u64)
        .unwrap_or(lo + 5);
    let len = lo + rng.below_u64(hi - lo + 1);
    (0..len).map(|_| rng.pick(&chars)).collect()
}

/// `inst` printed with each whitespace run drawn from `run`.
pub(super) fn write(inst: &Inst, run: &mut dyn FnMut() -> String) -> String {
    let mut out = run();
    put(inst, run, &mut out);
    out.push_str(&run());
    out
}

fn put(inst: &Inst, run: &mut dyn FnMut() -> String, out: &mut String) {
    let (open, close, len) = match inst {
        Inst::Atom(value) => {
            out.push_str(&serde_json::to_string(value).unwrap_or_default());
            return;
        }
        Inst::Obj(members) => ('{', '}', members.len()),
        Inst::Arr(items) => ('[', ']', items.len()),
    };
    out.push(open);
    out.push_str(&run());
    for i in 0..len {
        if i > 0 {
            out.push_str(&run());
            out.push(',');
            out.push_str(&run());
        }
        match inst {
            Inst::Obj(members) => {
                let (key, value) = &members[i];
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push_str(&run());
                out.push(':');
                out.push_str(&run());
                put(value, run, out);
            }
            Inst::Arr(items) => put(&items[i], run, out),
            Inst::Atom(_) => {}
        }
    }
    if len > 0 {
        out.push_str(&run());
    }
    out.push(close);
}

/// `inst` as a `serde_json` value.
pub(super) fn to_value(inst: &Inst) -> Value {
    match inst {
        Inst::Atom(value) => value.clone(),
        Inst::Obj(members) => Value::Object(
            members
                .iter()
                .map(|(k, v)| (k.clone(), to_value(v)))
                .collect(),
        ),
        Inst::Arr(items) => Value::Array(items.iter().map(to_value).collect()),
    }
}

/// True when some `Atom` of `inst` is an object or array.
pub(super) fn has_container_atom(inst: &Inst) -> bool {
    match inst {
        Inst::Atom(value) => value.is_object() || value.is_array(),
        Inst::Obj(members) => members.iter().any(|(_, v)| has_container_atom(v)),
        Inst::Arr(items) => items.iter().any(has_container_atom),
    }
}

/// The JSON Schema types a root value of `schema` can have.
fn types(schema: &Value) -> Vec<String> {
    let map = schema.as_object().cloned().unwrap_or_default();
    let mut out: Vec<String> = Vec::new();
    if map.get("nullable") == Some(&json!(true)) {
        out.push("null".into());
    }
    let members: Vec<Value> = match (map.get("enum"), map.get("const")) {
        (Some(Value::Array(m)), _) => m.clone(),
        (_, Some(c)) => vec![c.clone()],
        _ => Vec::new(),
    };
    for m in &members {
        out.push(type_of(m).into());
    }
    if let Some(Value::Array(subs)) = map.get("anyOf") {
        subs.iter().for_each(|s| out.extend(types(s)));
    }
    match map.get("type") {
        Some(Value::String(t)) => out.push(t.clone()),
        Some(Value::Array(ts)) => {
            out.extend(ts.iter().filter_map(|t| t.as_str().map(String::from)))
        }
        _ => {}
    }
    if out.iter().any(|t| t == "number") {
        out.push("integer".into());
    }
    out
}

fn type_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// A compact text that `schema` rejects, made by mutating the valid root `inst`.
///
/// The mutation drops a required key, adds an unknown key, swaps an `enum`
/// member for a non-member, or flips the root type. Trailing text is the fallback.
pub(super) fn mutate(rng: &mut Lcg, schema: &Value, inst: &Inst) -> String {
    let map = schema.as_object().cloned().unwrap_or_default();
    let compact = |i: &Inst| serde_json::to_string(&to_value(i)).unwrap_or_default();
    if let Inst::Obj(members) = inst
        && map.get("type") == Some(&json!("object"))
    {
        let required = map.get("required").cloned().unwrap_or(json!([]));
        let present: Vec<usize> = (0..members.len())
            .filter(|&i| {
                required
                    .as_array()
                    .is_some_and(|r| r.contains(&json!(members[i].0)))
            })
            .collect();
        let mut out = members.clone();
        if !present.is_empty() && rng.below_u64(2) == 0 {
            out.remove(present[rng.below_u64(present.len() as u64) as usize]);
        } else {
            out.push(("zz_unknown".into(), Inst::Atom(Value::Null)));
        }
        return compact(&Inst::Obj(out));
    }
    if (map.contains_key("enum") || map.contains_key("const")) && !map.contains_key("nullable") {
        return compact(&Inst::Atom(json!("zz-not-a-member")));
    }
    let allowed = types(schema);
    let flips = [
        ("null", json!(null)),
        ("boolean", json!(true)),
        ("integer", json!(7)),
        ("number", json!(1.5)),
        ("string", json!("zz")),
        ("object", json!({})),
        ("array", json!([])),
    ];
    let options: Vec<&Value> = flips
        .iter()
        .filter(|(t, _)| !allowed.iter().any(|a| a == t))
        .map(|(_, v)| v)
        .collect();
    if options.is_empty() {
        return format!("{}x", compact(inst));
    }
    let pick = options[rng.below_u64(options.len() as u64) as usize];
    serde_json::to_string(pick).unwrap_or_default()
}
