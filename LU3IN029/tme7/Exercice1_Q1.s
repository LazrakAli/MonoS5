.data

ch:	.asciiz "l exemple d'exemple\n"

.text
#prologue
	addiu	$29,$29,-4
main:
# i=0
	sw	$0,0($29)
# printf("%s",ch);	
	lui	$8,0x1001
	ori	$4,$8,0
	ori	$2,$0,4
	syscall

	lui	$8,0x1001
	ori	$15,$0,0x20
	lw	$14,0($29)
while:
#chargement chi
	lb	$9,0($8)
	
# ch[i] != ’\0’
	beq	$9,$0,endwhile

# if (ch[i] >= ’a’ && ch[i] <= ’z’)
	ori	$12,$0,0x61
    	slt     $12, $9, $12
    	bgtz    $12, endif
		
	ori	$12,$0,0x7A
    	slt     $12, $12,$9
    	bgtz    $12, endif
#ch[i] = ch[i] - 0x20;
	sub	$9,$9,$15
	sb	$9,0($8)

endif:
	addiu	$14,$14,1
	addiu	$8,$8,1
	j while

endwhile:
# printf("%s", ch)
	lui	$8,0x1001
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
# exit()
	ori 	$2,$0,10
	syscall