# BrailleLM — baseline vs native, side by side

Held-out test set: 120 sentences, same inputs to both models, greedy decoding.

| # | Reference | Baseline (raw braille, base vocab) | Native (braille tokens + BKFT + LoRA) |
|---:|---|---|---|
| 1 | The gardener started a wet umbrella before the storm. |  The fisherman prepared the stone bridge before the storm. | OK The gardener started a wet umbrella before the storm. |
| 2 | A dancer watched a bright lantern along the coast. |  The fisherman closed the stone bridge before the storm. | OK A dancer watched a bright lantern along the coast. |
| 3 | The librarian recorded a wet umbrella near the river. |  The captain opened the glass window before the storm. | OK The librarian recorded a wet umbrella near the river. |
| 4 | A student watched the empty basket beside the harbour. |  The fisherman carried a silver coin during the festival. | OK A student watched the empty basket beside the harbour. |
| 5 | The captain examined the iron gate near the river. |  The fisherman closed the stone bridge before the storm. | OK The captain examined the iron gate near the river. |
| 6 | The tailor counted the iron gate before the storm. |  The fisherman closed the stone bridge before the storm. | OK The tailor counted the iron gate before the storm. |
| 7 | The pilot examined a heavy box above the village. |  The fisherman closed the glass window before the storm. | OK The pilot examined a heavy box above the village. |
| 8 | The merchant arranged the stone bridge under the old tree. |  The captain recorded a silver coin in the morning. | OK The merchant arranged the stone bridge under the old tree. |
| 9 | The teacher finished ripe red apples under the old tree. |  The captain recorded a silver coin in the morning. | OK The teacher finished ripe red apples under the old tree. |
| 10 | The librarian discovered the tall ladder above the village. |  The captain recorded a silver coin in the morning. | OK The librarian discovered the tall ladder above the village. |
| 11 | A painter studied some fresh bread above the village. |  The fisherman closed the stone bridge before the storm. | OK A painter studied some fresh bread above the village. |
| 12 | The artist measured warm woollen blankets after the rain. |  The captain recorded a silver coin in the morning. | OK The artist measured warm woollen blankets after the rain. |
| 13 | The artist started the broken clock along the coast. |  The fisherman closed the stone bridge before the storm. | OK The artist started the broken clock along the coast. |
| 14 | My neighbour mentioned the green field in the morning. |  The fisherman prepared the stone bridge before the storm. | OK My neighbour mentioned the green field in the morning. |
| 15 | My neighbour explained some fresh bread near the river. |  The fisherman prepared the stone bridge before the storm. | OK My neighbour explained some fresh bread near the river. |
| 16 | The merchant discovered ripe red apples along the coast. |  The fisherman closed the stone bridge before the storm. | OK The merchant discovered ripe red apples along the coast. |
| 17 | The traveller recorded ripe red apples throughout the winter. |  The captain opened the glass window before the storm. | OK The traveller recorded ripe red apples throughout the winter. |
| 18 | The traveller discovered the iron gate near the river. |  The fisherman prepared the stone bridge before the storm. | OK The traveller discovered the iron gate near the river. |
| 19 | A painter arranged the small boat inside the museum. |  The fisherman closed the stone bridge before the storm. | OK A painter arranged the small boat inside the museum. |
| 20 | The pilot collected a wet umbrella after the rain. |  The fisherman closed the stone bridge before the storm. | OK The pilot collected a wet umbrella after the rain. |
| 21 | The librarian planted warm woollen blankets along the coast. |  The captain recorded a silver coin in the morning. | OK The librarian planted warm woollen blankets along the coast. |
| 22 | The doctor counted many wooden chairs across the valley. |  The tailor closed the stone bridge before the storm. | OK The doctor counted many wooden chairs across the valley. |
| 23 | A student followed the iron gate beneath the bridge. |  The fisherman closed the stone bridge before the storm. | OK A student followed the iron gate beneath the bridge. |
| 24 | A dancer discovered the stone bridge in the morning. |  The fisherman closed the stone bridge before the storm. | OK A dancer discovered the stone bridge in the morning. |
| 25 | The farmer mentioned some fresh bread along the coast. |  The fisherman prepared the stone bridge before the storm. | OK The farmer mentioned some fresh bread along the coast. |
| 26 | The doctor arranged several old books along the coast. |  The fisherman prepared the stone bridge before the storm. | OK The doctor arranged several old books along the coast. |
| 27 | The hunter arranged many wooden chairs during the festival. |  The captain recorded a silver coin in the morning. | OK The hunter arranged many wooden chairs during the festival. |
| 28 | A scientist planted several old books near the river. |  The fisherman prepared the stone bridge before the storm. | OK A scientist planted several old books near the river. |
| 29 | The tailor started the quiet garden during the festival. |  The tailor closed the stone bridge before the storm. | OK The tailor started the quiet garden during the festival. |
| 30 | A shopkeeper polished the empty basket near the river. |  The fisherman prepared the stone bridge before the storm. | OK A shopkeeper polished the empty basket near the river. |
| 31 | The nurse measured the quiet garden near the river. |  The captain opened the glass window before the storm. | OK The nurse measured the quiet garden near the river. |
| 32 | The teacher repaired some fresh bread near the river. |  The captain opened the glass window before the storm. | OK The teacher repaired some fresh bread near the river. |
| 33 | A scientist remembered the small boat behind the barn. |  The fisherman closed the stone bridge before the storm. | OK A scientist remembered the small boat behind the barn. |
| 34 | The doctor finished the green field near the river. |  The captain opened the glass window before the storm. | OK The doctor finished the green field near the river. |
| 35 | A dancer noticed several old books across the valley. |  The fisherman prepared the stone bridge before the storm. | OK A dancer noticed several old books across the valley. |
| 36 | The writer polished some fresh bread inside the museum. |  The captain prepared a silver coin in the morning. | OK The writer polished some fresh bread inside the museum. |
| 37 | The captain followed the tall ladder in the morning. |  The fisherman closed the stone bridge before the storm. | OK The captain followed the tall ladder in the morning. |
| 38 | The gardener arranged some fresh bread inside the museum. |  The captain recorded a silver coin in the morning. | OK The gardener arranged some fresh bread inside the museum. |
| 39 | The writer protected the stone bridge throughout the winter. |  The captain recorded a silver coin in the morning. | OK The writer protected the stone bridge throughout the winter. |
| 40 | A painter finished a silver coin after the rain. |  The fisherman closed the stone bridge before the storm. | OK A painter finished a silver coin after the rain. |
| 41 | A dancer started the iron gate inside the museum. |  The fisherman closed the glass window before the storm. | OK A dancer started the iron gate inside the museum. |
| 42 | The hunter opened ripe red apples above the village. |  The fisherman closed the stone bridge before the storm. | OK The hunter opened ripe red apples above the village. |
| 43 | The traveller measured the empty basket above the village. |  The captain recorded a silver coin in the morning. | OK The traveller measured the empty basket above the village. |
| 44 | A sailor carried some fresh bread before the storm. |  The fisherman closed the stone bridge before the storm. | OK A sailor carried some fresh bread before the storm. |
| 45 | A painter followed some fresh bread near the river. |  The fisherman closed the stone bridge before the storm. | OK A painter followed some fresh bread near the river. |
| 46 | The fisherman finished the tall ladder across the valley. |  The captain recorded a silver coin in the morning. | OK The fisherman finished the tall ladder across the valley. |
| 47 | The merchant repaired the glass window before the storm. |  The tailor closed the stone bridge before the storm. | OK The merchant repaired the glass window before the storm. |
| 48 | A child explained several old books during the festival. |  The fisherman closed the stone bridge before the storm. | OK A child explained several old books during the festival. |
| 49 | The librarian started the broken clock before the storm. |  The captain recorded a silver coin in the morning. | OK The librarian started the broken clock before the storm. |
| 50 | The engineer polished a wet umbrella under the old tree. |  The fisherman closed the stone bridge before the storm. | OK The engineer polished a wet umbrella under the old tree. |
| 51 | The librarian arranged the glass window in the morning. |  The captain prepared a silver coin in the morning. | OK The librarian arranged the glass window in the morning. |
| 52 | The merchant counted the empty basket near the river. |  The captain opened the glass window before the storm. | OK The merchant counted the empty basket near the river. |
| 53 | The young girl remembered a wet umbrella across the valley. |  The captain recorded a silver coin in the morning. | OK The young girl remembered a wet umbrella across the valley. |
| 54 | The hunter mentioned the glass window beneath the bridge. |  The captain recorded a silver coin in the morning. | OK The hunter mentioned the glass window beneath the bridge. |
| 55 | The tailor repaired several old books throughout the winter. |  The captain recorded a silver coin in the morning. | OK The tailor repaired several old books throughout the winter. |
| 56 | The tailor closed the stone bridge along the coast. |  The captain opened the glass window before the storm. | OK The tailor closed the stone bridge along the coast. |
| 57 | A dancer watched the small boat across the valley. |  The fisherman closed the stone bridge before the storm. | OK A dancer watched the small boat across the valley. |
| 58 | The merchant followed the narrow road beside the harbour. |  The captain recorded a silver coin in the morning. | OK The merchant followed the narrow road beside the harbour. |
| 59 | A dancer folded the glass window behind the barn. |  The fisherman closed the glass window before the storm. | OK A dancer folded the glass window behind the barn. |
| 60 | The librarian noticed the quiet garden throughout the winter. |  The captain recorded a silver coin in the morning. | OK The librarian noticed the quiet garden throughout the winter. |
| 61 | The traveller examined ripe red apples near the river. |  The fisherman prepared the stone bridge before the storm. | OK The traveller examined ripe red apples near the river. |
| 62 | The farmer discovered several old books around the market. |  The captain recorded a silver coin in the morning. | OK The farmer discovered several old books around the market. |
| 63 | The captain opened a paper map above the village. |  The fisherman closed the glass window before the storm. | OK The captain opened a paper map above the village. |
| 64 | A sailor described many wooden chairs along the coast. |  The fisherman closed the stone bridge before the storm. | OK A sailor described many wooden chairs along the coast. |
| 65 | The artist repaired ripe red apples above the village. |  The fisherman prepared the stone bridge before the storm. | OK The artist repaired ripe red apples above the village. |
| 66 | The gardener counted the narrow road across the valley. |  The captain prepared a silver coin in the morning. | OK The gardener counted the narrow road across the valley. |
| 67 | A shopkeeper protected the stone bridge above the village. |  The captain recorded a silver coin in the morning. | OK A shopkeeper protected the stone bridge above the village. |
| 68 | The teacher described the narrow road inside the museum. |  The fisherman closed the stone bridge before the storm. | OK The teacher described the narrow road inside the museum. |
| 69 | The traveller arranged a heavy box in the morning. |  The fisherman closed the stone bridge before the storm. | OK The traveller arranged a heavy box in the morning. |
| 70 | The captain studied the green field near the river. |  The captain opened the glass window before the storm. | OK The captain studied the green field near the river. |
| 71 | A student started the empty basket beneath the bridge. |  The fisherman carried the stone bridge before the storm. | OK A student started the empty basket beneath the bridge. |
| 72 | The traveller delivered the broken clock inside the museum. |  The captain recorded a silver coin in the morning. | OK The traveller delivered the broken clock inside the museum. |
| 73 | The pilot studied a bright lantern along the coast. |  The captain opened the glass window before the storm. | OK The pilot studied a bright lantern along the coast. |
| 74 | The pilot repaired the quiet garden along the coast. |  The fisherman closed the stone bridge before the storm. | OK The pilot repaired the quiet garden along the coast. |
| 75 | A carpenter watched some fresh bread during the festival. |  The captain recorded a silver coin in the morning. | OK A carpenter watched some fresh bread during the festival. |
| 76 | A musician protected the tall ladder during the festival. |  The captain recorded a silver coin in the morning. | OK A musician protected the tall ladder during the festival. |
| 77 | The traveller opened the empty basket around the market. |  The tailor closed the stone bridge before the storm. | OK The traveller opened the empty basket around the market. |
| 78 | The librarian carried a wet umbrella inside the museum. |  The captain prepared a silver coin in the morning. | OK The librarian carried a wet umbrella inside the museum. |
| 79 | A musician repaired the small boat behind the barn. |  The fisherman closed the stone bridge before the storm. | OK A musician repaired the small boat behind the barn. |
| 80 | The baker followed the quiet garden along the coast. |  The fisherman closed the stone bridge before the storm. | OK The baker followed the quiet garden along the coast. |
| 81 | The doctor examined the small boat above the village. |  The fisherman prepared the stone bridge before the storm. | OK The doctor examined the small boat above the village. |
| 82 | A painter folded a heavy box along the coast. |  The fisherman closed the stone bridge before the storm. | OK A painter folded a heavy box along the coast. |
| 83 | The farmer remembered the small boat during the festival. |  The captain recorded a silver coin in the morning. | OK The farmer remembered the small boat during the festival. |
| 84 | The merchant carried the glass window around the market. |  The tailor closed the stone bridge before the storm. | OK The merchant carried the glass window around the market. |
| 85 | The doctor sketched the stone bridge near the river. |  The fisherman closed the stone bridge before the storm. | OK The doctor sketched the stone bridge near the river. |
| 86 | The fisherman measured some fresh bread along the coast. |  The tailor closed the stone bridge before the storm. | OK The fisherman measured some fresh bread along the coast. |
| 87 | The gardener discovered a wet umbrella behind the barn. |  The fisherman prepared the stone bridge before the storm. | OK The gardener discovered a wet umbrella behind the barn. |
| 88 | A scientist noticed a heavy box behind the barn. |  The fisherman closed the stone bridge before the storm. | OK A scientist noticed a heavy box behind the barn. |
| 89 | The fisherman opened the stone bridge throughout the winter. |  The captain recorded a silver coin in the morning. | OK The fisherman opened the stone bridge throughout the winter. |
| 90 | The baker described the broken clock behind the barn. |  The captain opened the glass window before the storm. | OK The baker described the broken clock behind the barn. |
| 91 | The weaver followed the small boat across the valley. |  The captain opened the glass window before the storm. | OK The weaver followed the small boat across the valley. |
| 92 | The old man recorded a bright lantern during the festival. |  The captain recorded a silver coin in the morning. | OK The old man recorded a bright lantern during the festival. |
| 93 | The captain recorded the quiet garden across the valley. |  The tailor closed the stone bridge before the storm. | OK The captain recorded the quiet garden across the valley. |
| 94 | The young girl started several old books during the festival. |  The captain recorded a silver coin in the morning. | OK The young girl started several old books during the festival. |
| 95 | The merchant examined warm woollen blankets beside the harbour. |  The tailor closed the stone bridge before the storm. | OK The merchant examined warm woollen blankets beside the harbour. |
| 96 | A sailor measured the stone bridge before the storm. |  The fisherman closed the stone bridge before the storm. | OK A sailor measured the stone bridge before the storm. |
| 97 | The doctor opened the small boat along the coast. |  The fisherman closed the glass window before the storm. | OK The doctor opened the small boat along the coast. |
| 98 | The gardener examined several old books beside the harbour. |  The captain recorded a silver coin in the morning. | OK The gardener examined several old books beside the harbour. |
| 99 | The artist closed a wet umbrella in the morning. |  The fisherman closed the stone bridge before the storm. | OK The artist closed a wet umbrella in the morning. |
| 100 | The farmer arranged ripe red apples in the morning. |  The captain opened the glass window before the storm. | OK The farmer arranged ripe red apples in the morning. |
| 101 | The tailor sketched many wooden chairs above the village. |  The captain recorded a silver coin in the morning. | OK The tailor sketched many wooden chairs above the village. |
| 102 | The hunter studied several old books after the rain. |  The fisherman closed the stone bridge before the storm. | OK The hunter studied several old books after the rain. |
| 103 | The merchant mentioned a heavy box inside the museum. |  The captain opened the glass window before the storm. | OK The merchant mentioned a heavy box inside the museum. |
| 104 | The weaver painted a paper map inside the museum. |  The fisherman closed the glass window before the storm. | OK The weaver painted a paper map inside the museum. |
| 105 | The teacher painted a paper map behind the barn. |  The fisherman closed the glass window before the storm. | OK The teacher painted a paper map behind the barn. |
| 106 | The librarian prepared a wet umbrella throughout the winter. |  The captain recorded a silver coin in the morning. | OK The librarian prepared a wet umbrella throughout the winter. |
| 107 | The farmer noticed many wooden chairs near the river. |  The captain opened the glass window before the storm. | OK The farmer noticed many wooden chairs near the river. |
| 108 | The engineer studied some fresh bread behind the barn. |  The fisherman prepared the stone bridge before the storm. | OK The engineer studied some fresh bread behind the barn. |
| 109 | The nurse prepared the glass window after the rain. |  The fisherman closed the stone bridge before the storm. | OK The nurse prepared the glass window after the rain. |
| 110 | A scientist examined the tall ladder beneath the bridge. |  The fisherman closed the stone bridge before the storm. | OK A scientist examined the tall ladder beneath the bridge. |
| 111 | The weaver repaired the narrow road under the old tree. |  The tailor closed the stone bridge before the storm. | OK The weaver repaired the narrow road under the old tree. |
| 112 | The librarian collected a heavy box before the storm. |  The captain opened the glass window before the storm. | OK The librarian collected a heavy box before the storm. |
| 113 | The weaver opened the quiet garden under the old tree. |  The fisherman closed the stone bridge before the storm. | OK The weaver opened the quiet garden under the old tree. |
| 114 | My neighbour painted a bright lantern inside the museum. |  The fisherman prepared the stone bridge before the storm. | OK My neighbour painted a bright lantern inside the museum. |
| 115 | A sailor counted the empty basket beneath the bridge. |  The fisherman closed the stone bridge before the storm. | OK A sailor counted the empty basket beneath the bridge. |
| 116 | The fisherman gathered the green field in the morning. |  The fisherman prepared the stone bridge before the storm. | OK The fisherman gathered the green field in the morning. |
| 117 | A dancer watched the empty basket in the morning. |  The fisherman closed the stone bridge before the storm. | OK A dancer watched the empty basket in the morning. |
| 118 | The captain gathered some fresh bread after the rain. |  The fisherman prepared the stone bridge before the storm. | OK The captain gathered some fresh bread after the rain. |
| 119 | The gardener started a bright lantern inside the museum. |  The captain prepared a silver coin in the morning. | OK The gardener started a bright lantern inside the museum. |
| 120 | A carpenter noticed the small boat around the market. |  The fisherman prepared the stone bridge before the storm. | OK A carpenter noticed the small boat around the market. |

Full braille inputs are in `data/test.jsonl`; raw predictions in `preds_baseline.tsv` / `preds_native.tsv`.