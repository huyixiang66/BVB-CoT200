import json
import tempfile
import unittest
from pathlib import Path
from bvb_v13.answer_text import parse,build


class AnswerOnly(unittest.TestCase):
    def test_counting_uses_all_render_palette_colors_without_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);q=dict(question='Count chairs and tables',objects=['chair','table'])
            packet=dict(counting_mode=True,colors={'chair':[1,0,0,1],'table':[0,1,0,1]})
            palettes={'chair':[[1,0,0,1],[0,0,1,1]],'table':[[0,1,0,1],[1,1,0,1]]}
            (out/'input.json').write_text(json.dumps(packet))
            (out/'qa-color-legend.json').write_text(json.dumps(dict(category_palette=palettes,instances='PRIVATE_INVENTORY',count=99)))
            text=build(out,q,'counting')
            self.assertIn(json.dumps(palettes,separators=(',',':')),text)
            self.assertIn('not only objects in its first palette color',text)
            self.assertNotIn('Category colors:',text);self.assertNotIn('PRIVATE_INVENTORY',text)
            self.assertNotIn('99',text);self.assertIn('No JSON',text)

    def test_counting_fails_closed_on_missing_or_invalid_palette(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);q=dict(objects=['chair']);(out/'input.json').write_text('{"counting_mode":true}')
            with self.assertRaises(FileNotFoundError):build(out,q,'counting')
            for palette in [{},{'table':[[1,0,0,1]]},{'chair':[]},{'chair':[[2,0,0,1]]}]:
                (out/'qa-color-legend.json').write_text(json.dumps(dict(category_palette=palette)))
                with self.assertRaises(ValueError):build(out,q,'counting')

    def test_regular_category_colors_and_zero_evidence_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);q=dict(objects=['chair']);packet=dict(counting_mode=False,colors={'chair':[1,0,0,1]})
            (out/'input.json').write_text(json.dumps(packet))
            self.assertIn('Category colors:',build(out,q,'counting'))
            packet.update(counting_mode=True,counting_allow_all_rejected=True,counting_zero_evidence_categories=['chair'])
            (out/'input.json').write_text(json.dumps(packet));(out/'qa-color-legend.json').write_text('{"category_palette":{}}')
            self.assertIn('Category palettes',build(out,q,'counting'))

    def test_choice_normalizes_locally_without_inventing_reason(self):
        for task in ['rel_dir','rel_dist','route']:
            r=parse('b',dict(options=['A. left','B. right']),task)
            self.assertEqual(r['option'],'B');self.assertEqual(r['answer'],'right')
            self.assertNotIn('reason',r);self.assertNotIn('confidence',r)

    def test_numeric_units_and_counts(self):
        for task,unit in [('abs_dist','m'),('object_size','cm'),('room_size','m²')]:
            r=parse('2.5',{},task);self.assertEqual(r['answer'],2.5);self.assertEqual(r['unit'],unit)
        self.assertIs(type(parse('3',{},'counting')['answer']),int)
        for text in ['3.5','-1','nan','1e9999','{"answer":3}','3 chairs']:
            with self.assertRaises(ValueError):parse(text,{},'counting')
        with self.assertRaises(ValueError):parse('1e9999',{},'abs_dist')
        with self.assertRaises(ValueError):parse('C',dict(options=['A. left','B. right']),'rel_dir')

    def test_more_has_no_provisional_answer(self):
        self.assertEqual(parse('NEED_MORE_VIEWS',{},'room_size'),dict(need_more_views=True))

    def test_prompt_excludes_private_fields_and_output_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            q=dict(question='How many chairs?',objects=['chair'],ground_truth='SECRET_GT',queried_object_ids=['PRIVATE_ID'])
            p=build(Path(tmp),q,'counting')
            self.assertNotIn('SECRET_GT',p);self.assertNotIn('PRIVATE_ID',p)
            self.assertIn('No JSON',p);self.assertNotIn('counts_by_category',p)


if __name__=='__main__':unittest.main()
