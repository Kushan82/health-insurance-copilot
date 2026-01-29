"""
Ingest master policy JSON documents into vector store
FIXED: Includes required source, start_char, end_char fields
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict, Any
from loguru import logger

from src.services.rag.chunker import Chunk
from src.services.rag.embedding_service import EmbeddingService
from src.services.rag.vector_store import VectorStore


def load_master_document(json_path: Path) -> Dict[str, Any]:
    """Load master JSON document"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {json_path}: {e}")
        return None


def create_chunks_from_master_doc(master_doc: Dict[str, Any], source_file: str) -> List[Chunk]:
    """Convert master document into optimized chunks"""
    chunks = []
    
    metadata = master_doc['policy_metadata']
    policy_name = metadata['policy_name']
    policy_uin = metadata.get('policy_uin', 'unknown')
    policy_lower = policy_name.lower()
    
    logger.info(f"Creating chunks for: {policy_name}")
    
    # CHUNK 1: Overview
    summary = master_doc.get('executive_summary', {})
    coverage = master_doc.get('coverage_summary', {})
    
    overview_text = f"""{policy_name} - Health Insurance Policy

UIN: {policy_uin}
Insurer: {metadata.get('insurer', 'Unknown')}

{summary.get('one_line_pitch', 'Comprehensive coverage')}

IDEAL FOR:
{chr(10).join('• ' + item for item in summary.get('ideal_for', []))}

NOT IDEAL FOR:
{chr(10).join('• ' + item for item in summary.get('not_ideal_for', []))}

KEY FEATURES:
{chr(10).join('• ' + item for item in summary.get('standout_features', []))}

LIMITATIONS:
{chr(10).join('• ' + item for item in summary.get('major_limitations', []))}

VALUE: {summary.get('overall_value_assessment', 'Not specified')}

SUM INSURED OPTIONS:
{chr(10).join(f"• {si.get('amount', '')} - {si.get('suitable_for', '')}" for si in coverage.get('sum_insured_options', []))}

ELIGIBILITY: {coverage.get('eligibility', {}).get('entry_age_adult', {}).get('min', 18)}-{coverage.get('eligibility', {}).get('entry_age_adult', {}).get('max', 65)} years
"""
    
    chunks.append(Chunk(
        text=overview_text.strip(),
        metadata={"filename": source_file, "policy": policy_lower, "section": "overview", "chunk_type": "master_document", "policy_uin": policy_uin},
        chunk_id=f"{policy_uin}_overview",
        source=source_file,
        start_char=0,
        end_char=len(overview_text)
    ))
    
    # CHUNK 2: Waiting Periods
    waiting = master_doc.get('waiting_periods', {})
    initial = waiting.get('initial_waiting_period', {})
    specific = waiting.get('specific_diseases_waiting', {})
    ped = waiting.get('pre_existing_diseases', {})
    
    waiting_text = f"""Waiting Periods - {policy_name}

SUMMARY: {waiting.get('summary', 'Standard waiting periods')}
INDUSTRY: {waiting.get('comparison_to_industry', 'Standard')}

INITIAL: {initial.get('duration', 'Not specified')}
Applies to: {initial.get('applies_to', 'All illnesses except accidents')}
Impact: {initial.get('user_impact', 'Coverage starts after period')}

SPECIFIC DISEASES: {specific.get('duration', 'Not specified')}
Conditions: {', '.join(specific.get('covered_conditions', [])[:6])}

PRE-EXISTING DISEASES: {ped.get('duration', 'Not specified')}
Definition: {ped.get('definition_simple', 'Diseases before policy')}
Must Declare: {'Yes' if ped.get('must_declare') else 'No'}
Impact: {ped.get('user_impact', 'Disclose honestly')}
Comparison: {ped.get('comparison_note', 'Standard')}

MATERNITY: {waiting.get('maternity_waiting', {}).get('duration', 'Not covered')}
"""
    
    chunks.append(Chunk(
        text=waiting_text.strip(),
        metadata={"filename": source_file, "policy": policy_lower, "section": "waiting_periods", "chunk_type": "master_document", "policy_uin": policy_uin},
        chunk_id=f"{policy_uin}_waiting",
        source=source_file,
        start_char=0,
        end_char=len(waiting_text)
    ))
    
    # CHUNK 3: Hospitalization
    hosp = master_doc.get('hospitalization_benefits', {})
    room = hosp.get('room_rent', {})
    
    hosp_text = f"""Hospitalization Benefits - {policy_name}

ROOM RENT: {room.get('type', 'Not specified')}
Limit: {room.get('limit', 'Not specified')}
ICU: {room.get('icu_coverage', 'Not specified')}
Impact: {room.get('impact_explanation', 'Affects claim amount')}

PRE-HOSPITALIZATION: {hosp.get('pre_hospitalization', {}).get('duration_text', 'Not specified')}
POST-HOSPITALIZATION: {hosp.get('post_hospitalization', {}).get('duration_text', 'Not specified')}

DAY CARE: {'Yes' if hosp.get('day_care_procedures', {}).get('covered') else 'No'}
Count: {hosp.get('day_care_procedures', {}).get('number_of_procedures_listed', 'Multiple')}
Examples: {', '.join(hosp.get('day_care_procedures', {}).get('examples', [])[:5])}

AMBULANCE: {hosp.get('ambulance_charges', {}).get('limit_per_hospitalization', 'Not specified')}
"""
    
    chunks.append(Chunk(
        text=hosp_text.strip(),
        metadata={"filename": source_file, "policy": policy_lower, "section": "hospitalization", "chunk_type": "master_document", "policy_uin": policy_uin},
        chunk_id=f"{policy_uin}_hospitalization",
        source=source_file,
        start_char=0,
        end_char=len(hosp_text)
    ))
    
    # CHUNK 4: Special Benefits
    special = master_doc.get('special_benefits', {})
    restore = special.get('restore_benefit', {})
    ncb = special.get('no_claim_bonus', {})
    maternity = special.get('maternity_coverage', {})
    
    special_text = f"""Special Benefits - {policy_name}

RESTORE BENEFIT: {'Yes' if restore.get('available') else 'No'}
Type: {restore.get('type', 'N/A')}
Amount: {restore.get('restoration_amount', 'N/A')}
Times: {restore.get('number_of_times', 'N/A')}
Impact: {restore.get('user_impact', '')}

NO CLAIM BONUS: {'Yes' if ncb.get('available') else 'No'}
Increment: {ncb.get('increment_per_year', 'N/A')}
Maximum: {ncb.get('maximum_bonus', 'N/A')}

MATERNITY: {'Yes' if maternity.get('covered') else 'No'}
Waiting: {maternity.get('waiting_period', 'N/A')}
Normal: {maternity.get('normal_delivery_limit', 'N/A')}
C-Section: {maternity.get('cesarean_delivery_limit', 'N/A')}

AYUSH: {'Yes' if special.get('ayush_treatment', {}).get('covered') else 'No'}
"""
    
    chunks.append(Chunk(
        text=special_text.strip(),
        metadata={"filename": source_file, "policy": policy_lower, "section": "special_benefits", "chunk_type": "master_document", "policy_uin": policy_uin},
        chunk_id=f"{policy_uin}_special",
        source=source_file,
        start_char=0,
        end_char=len(special_text)
    ))
    
    # CHUNK 5: Exclusions
    exclusions = master_doc.get('exclusions', {})
    
    excl_text = f"""Exclusions - {policy_name}

{exclusions.get('summary', 'Standard exclusions apply')}

CRITICAL EXCLUSIONS:
"""
    for excl in exclusions.get('critical_exclusions_explained', [])[:5]:
        excl_text += f"\n{excl.get('exclusion', '')}: {excl.get('simple_explanation', '')}"
        excl_text += f"\nExamples: {', '.join(excl.get('examples', [])[:3])}"
    
    excl_text += "\n\nPERMANENT EXCLUSIONS:\n"
    for excl in exclusions.get('permanent_exclusions', [])[:8]:
        excl_text += f"• {excl}\n"
    
    chunks.append(Chunk(
        text=excl_text.strip(),
        metadata={"filename": source_file, "policy": policy_lower, "section": "exclusions", "chunk_type": "master_document", "policy_uin": policy_uin},
        chunk_id=f"{policy_uin}_exclusions",
        source=source_file,
        start_char=0,
        end_char=len(excl_text)
    ))
    
    # CHUNK 6: Sub-limits
    sublimits = master_doc.get('sub_limits_and_caps', {})
    
    sublim_text = f"""Sub-limits and Caps - {policy_name}

{sublimits.get('summary', 'Check policy for limits')}

ROOM RENT CAP: {'Yes' if sublimits.get('room_rent_cap', {}).get('applicable') else 'No'}
Cap: {sublimits.get('room_rent_cap', {}).get('cap', 'No cap')}

PROCEDURE LIMITS:
Cataract: {sublimits.get('cataract', {}).get('limit', 'No limit')}
Joint Replacement: {sublimits.get('joint_replacement', {}).get('limit', 'No limit')}
Hernia: {sublimits.get('hernia', {}).get('limit', 'No limit')}

COPAY: {'Yes - ' + sublimits.get('copay_clauses', {}).get('copay_percentage', '') if sublimits.get('copay_clauses', {}).get('applicable') else 'No'}
DEDUCTIBLE: {'Yes - ' + sublimits.get('deductible', {}).get('amount', '') if sublimits.get('deductible', {}).get('applicable') else 'No'}
"""
    
    chunks.append(Chunk(
        text=sublim_text.strip(),
        metadata={"filename": source_file, "policy": policy_lower, "section": "sublimits", "chunk_type": "master_document", "policy_uin": policy_uin},
        chunk_id=f"{policy_uin}_sublimits",
        source=source_file,
        start_char=0,
        end_char=len(sublim_text)
    ))
    
    # CHUNK 7: Comparison
    comparison = master_doc.get('policy_comparison_dimensions', {})
    guidance = master_doc.get('chatbot_guidance_notes', {})
    
    comp_text = f"""Policy Comparison - {policy_name}

SCORES:
Coverage: {comparison.get('comprehensive_coverage_score', 'N/A')}/10
Waiting: {comparison.get('waiting_periods_score', 'N/A')}/10
Claims: {comparison.get('claim_settlement_score', 'N/A')}/10
Value: {comparison.get('value_for_money_score', 'N/A')}/10

BEST FOR:
Young Individuals: {comparison.get('best_for_comparison', {}).get('young_individuals', 'Check features')}
Families: {comparison.get('best_for_comparison', {}).get('young_families', 'Check maternity')}
Seniors: {comparison.get('best_for_comparison', {}).get('senior_citizens', 'Check age limits')}

RECOMMENDATION: {guidance.get('recommendation_logic', 'Based on user needs')}

TALKING POINTS:
{chr(10).join('• ' + point for point in guidance.get('comparison_talking_points', [])[:5])}
"""
    
    chunks.append(Chunk(
        text=comp_text.strip(),
        metadata={"filename": source_file, "policy": policy_lower, "section": "comparison", "chunk_type": "master_document", "policy_uin": policy_uin},
        chunk_id=f"{policy_uin}_comparison",
        source=source_file,
        start_char=0,
        end_char=len(comp_text)
    ))
    
    logger.info(f"✅ Created {len(chunks)} chunks for {policy_name}")
    return chunks


def ingest_all_master_documents(reset_vector_store: bool = True):
    """Main ingestion function"""
    print("\n" + "="*60)
    print("  🚀 MASTER DOCUMENT INGESTION")
    print("="*60)
    
    logger.info("Initializing services...")
    embedding_service = EmbeddingService()
    vector_store = VectorStore(reset=reset_vector_store)
    
    if reset_vector_store:
        print("✅ Vector store reset - starting fresh")
    
    # Find master JSON files
    master_docs_dir = Path("data/raw/policies")
    subdirs = ['hdfc_ergo', 'bajaj', 'care_health', 'tata_aig']
    
    json_files = []
    for subdir in subdirs:
        subdir_path = master_docs_dir / subdir
        if subdir_path.exists():
            json_files.extend(list(subdir_path.glob("*_master.json")))
    
    if not json_files:
        logger.error(f"❌ No master JSON files found")
        return
    
    print(f"\n📄 Found {len(json_files)} master documents:")
    for json_file in json_files:
        print(f"   • {json_file.name}")
    
    total_chunks = 0
    successful = 0
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n{'─'*60}")
        print(f"Processing {i}/{len(json_files)}: {json_file.name}")
        print(f"{'─'*60}")
        
        try:
            master_doc = load_master_document(json_file)
            if not master_doc:
                continue
            
            policy_name = master_doc['policy_metadata']['policy_name']
            logger.info(f"Loaded: {policy_name}")
            
            print(f"   ✂️  Creating chunks...")
            chunks = create_chunks_from_master_doc(master_doc, json_file.name)
            
            print(f"   🔤 Generating embeddings...")
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = embedding_service.embed_batch(chunk_texts)
            
            print(f"   💾 Adding to vector store...")
            vector_store.add_chunks(chunks, embeddings)
            
            total_chunks += len(chunks)
            successful += 1
            
            print(f"   ✅ Successfully ingested {policy_name} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("  📊 INGESTION SUMMARY")
    print("="*60)
    print(f"\n✅ Successfully processed: {successful}/{len(json_files)} policies")
    print(f"📦 Total chunks ingested: {total_chunks}")
    print(f"📊 Average chunks per policy: {total_chunks/max(successful, 1):.1f}")
    
    stats = vector_store.get_stats()
    print(f"\n📈 Vector Store Statistics:")
    print(f"   • Total chunks: {stats.get('total_chunks', 0)}")
    print(f"   • Unique sources: {stats.get('unique_sources', 0)}")
    
    print("\n" + "="*60)
    print("  ✅ INGESTION COMPLETE!")
    print("="*60)
    print("\n📍 Next: python scripts/test_rag_complete.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest master policy documents")
    parser.add_argument('--reset', action='store_true', default=True, help='Reset vector store (default)')
    parser.add_argument('--no-reset', dest='reset', action='store_false', help='Append to existing')
    
    args = parser.parse_args()
    ingest_all_master_documents(reset_vector_store=args.reset)
